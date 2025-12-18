# Authentication System Quick Start Guide

**Version**: 1.0 | **Date**: 2024-12-18
**Prerequisites**: Node.js 18+, PostgreSQL 14+, Docusaurus 3.9.2

This guide will help you set up the authentication system for the Humanoid Robotics Lab Guide platform using better-auth.

## Overview

The authentication system consists of:
- **better-auth**: Modern authentication library with secure defaults
- **PostgreSQL**: Database for user data, progress tracking, and bookmarks
- **Express.js API**: Separate backend server for authentication
- **Docusaurus Plugin**: Custom integration for the documentation site
- **Prisma**: Type-safe ORM for database operations

## Prerequisites

### Required Software
- Node.js 18+ and npm
- PostgreSQL 14+ (or managed PostgreSQL service)
- Git
- Code editor (VS Code recommended)

### Development Environment
- Docusaurus 3.9.2 project structure
- TypeScript knowledge
- React development experience

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Docusaurus    │    │   Express.js     │    │   PostgreSQL    │
│   Frontend      │◄──►│   API Server     │◄──►│   Database      │
│   (React SPA)   │    │   /api/auth/*    │    │   (Users,       │
│   Client Routes │    │   better-auth    │    │    Sessions)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Quick Start (5 Minutes)

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-org/humanoid-robotics.git
cd humanoid-robotics

# Install dependencies
npm install

# Set up environment variables
cp .env.example .env.local
```

### 2. Database Setup

```bash
# Install PostgreSQL (if not already installed)
# On macOS: brew install postgresql
# On Ubuntu: sudo apt-get install postgresql postgresql-contrib

# Create database
createdb humanoid_robotics

# Set up Prisma
cd api
npx prisma generate
npx prisma db push

# Seed database (optional)
npx prisma db seed
```

### 3. API Server Configuration

Create `api/.env`:
```env
DATABASE_URL="postgresql://username:password@localhost:5432/humanoid_robotics"
BETTER_AUTH_SECRET="your-super-secure-256-bit-secret-key-here"
BETTER_AUTH_URL="http://localhost:4000"
FRONTEND_URL="http://localhost:3000"

# Email for password reset (optional)
SMTP_HOST="smtp.gmail.com"
SMTP_PORT=587
SMTP_USER="your-email@gmail.com"
SMTP_PASSWORD="your-app-password"
```

Start the API server:
```bash
cd api
npm run dev
# API server running on http://localhost:4000
```

### 4. Frontend Integration

Update `docusaurus.config.js`:
```javascript
const config = {
  // ... existing config
  plugins: [
    // ... existing plugins
    [require.resolve('./src/plugins/auth-plugin.js'), {
      apiUrl: process.env.AUTH_API_URL || 'http://localhost:4000/api/auth',
    }],
  ],
};
```

Start the frontend:
```bash
npm run start
# Docusaurus running on http://localhost:3000
```

### 5. Test Authentication

Visit `http://localhost:3000/auth/login` to test:
- User registration
- Login/logout
- Profile management

## Detailed Setup

### API Server Setup

#### 1. Initialize API Project

```bash
mkdir api
cd api
npm init -y
npm install express better-auth @better-auth/postgres prisma @prisma/client
npm install -D typescript @types/node ts-node nodemon
```

#### 2. Create better-auth Configuration

```typescript
// api/src/lib/auth.ts
import { betterAuth } from "better-auth";
import { postgresAdapter } from "@better-auth/postgres";
import { PrismaClient } from "@prisma/client";

const prisma = new PrismaClient();

export const auth = betterAuth({
  database: postgresAdapter(prisma),
  emailAndPassword: {
    enabled: true,
    requireEmailVerification: true,
  },
  session: {
    expiresIn: 60 * 60 * 24 * 7, // 7 days
  },
  socialProviders: {
    // Add social providers here if needed
  },
});
```

#### 3. Create Express Server

```typescript
// api/src/app.ts
import express from "express";
import cors from "cors";
import { auth } from "./lib/auth";

const app = express();

app.use(cors({
  origin: process.env.FRONTEND_URL,
  credentials: true,
}));

app.use(express.json());

// better-auth routes
app.use("/api/auth", auth.handler);

app.listen(4000, () => {
  console.log("Auth server running on http://localhost:4000");
});
```

#### 4. Database Schema

```prisma
// api/prisma/schema.prisma
generator client {
  provider = "prisma-client-js"
}

datasource db {
  provider = "postgresql"
  url      = env("DATABASE_URL")
}

model User {
  id            String    @id @default(cuid())
  email         String    @unique
  emailVerified Boolean   @default(false)
  name          String?
  image         String?
  createdAt     DateTime  @default(now())
  updatedAt     DateTime  @updatedAt

  accounts          Account[]
  sessions          Session[]
  verificationTokens VerificationToken[]
  userProgress      UserProgress[]
  bookmarks         Bookmark[]

  @@map("users")
}

// Add other better-auth models and custom models as shown in data-model.md
```

### Frontend Integration

#### 1. Create Auth Plugin

```javascript
// src/plugins/auth-plugin.js
module.exports = function authPlugin(context, options) {
  return {
    name: 'auth-plugin',
    getClientModules() {
      return [require.resolve('./components/Auth/AuthProvider')];
    },
    configureWebpack() {
      return {
        resolve: {
          alias: {
            '@auth': require.resolve('./lib/auth'),
            '@api': require.resolve('./lib/api'),
          },
        },
      };
    },
  };
};
```

#### 2. Create Auth Context

```typescript
// src/components/Auth/AuthProvider.tsx
import React, { createContext, useContext, useState, useEffect } from 'react';

interface AuthContextType {
  user: any | null;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, name: string) => Promise<void>;
  logout: () => Promise<void>;
  loading: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    checkAuth();
  }, []);

  const checkAuth = async () => {
    try {
      const response = await fetch('/api/auth/session');
      const data = await response.json();
      setUser(data.user);
    } catch (error) {
      setUser(null);
    } finally {
      setLoading(false);
    }
  };

  const login = async (email: string, password: string) => {
    const response = await fetch('/api/auth/sign-in', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });
    const data = await response.json();
    setUser(data.user);
  };

  const register = async (email: string, password: string, name: string) => {
    const response = await fetch('/api/auth/sign-up', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password, name }),
    });
    const data = await response.json();
    setUser(data.user);
  };

  const logout = async () => {
    await fetch('/api/auth/sign-out', { method: 'POST' });
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ user, login, register, logout, loading }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  return context;
};
```

#### 3. Create Login Form

```typescript
// src/components/Auth/LoginForm.tsx
import React, { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { useAuth } from './AuthProvider';
import toast from 'react-hot-toast';

const loginSchema = z.object({
  email: z.string().email('Invalid email address'),
  password: z.string().min(1, 'Password is required'),
});

type LoginFormData = z.infer<typeof loginSchema>;

export const LoginForm: React.FC = () => {
  const { login } = useAuth();
  const [loading, setLoading] = useState(false);

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<LoginFormData>({
    resolver: zodResolver(loginSchema),
  });

  const onSubmit = async (data: LoginFormData) => {
    setLoading(true);
    try {
      await login(data.email, data.password);
      toast.success('Successfully logged in!');
    } catch (error) {
      toast.error('Login failed. Please check your credentials.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)} className="auth-form">
      <h2>Login</h2>

      <div className="form-group">
        <label>Email</label>
        <input
          type="email"
          {...register('email')}
          className={errors.email ? 'error' : ''}
        />
        {errors.email && <span className="error-message">{errors.email.message}</span>}
      </div>

      <div className="form-group">
        <label>Password</label>
        <input
          type="password"
          {...register('password')}
          className={errors.password ? 'error' : ''}
        />
        {errors.password && <span className="error-message">{errors.password.message}</span>}
      </div>

      <button type="submit" disabled={loading} className="auth-button">
        {loading ? 'Logging in...' : 'Login'}
      </button>
    </form>
  );
};
```

## Features

### Authentication
- ✅ Email/password authentication
- ✅ Password reset via email
- ✅ Session management with secure cookies
- ✅ Rate limiting for security
- ✅ Email verification

### Progress Tracking
- ✅ Track learning progress per content page
- ✅ Completion percentages
- ✅ Last accessed timestamps
- ✅ Progress analytics

### Bookmarks
- ✅ Bookmark content pages
- ✅ Custom titles and descriptions
- ✅ Searchable bookmark library
- ✅ Export bookmarks

### User Profile
- ✅ Profile management
- ✅ Avatar upload
- ✅ Account settings
- ✅ Email preferences

## Configuration Options

### Security Settings

```typescript
// Enhanced security configuration
export const auth = betterAuth({
  session: {
    expiresIn: 60 * 60 * 24 * 7, // 7 days
    cookieCache: {
      enabled: true,
      maxAge: 5 * 60, // 5 minutes
    },
    cookieAttributes: {
      secure: process.env.NODE_ENV === 'production',
      httpOnly: true,
      sameSite: 'lax',
    },
  },
  security: {
    csrfProtection: {
      enabled: true,
      trustedOrigins: [process.env.FRONTEND_URL],
    },
    rateLimit: {
      enabled: true,
      window: 60 * 1000, // 1 minute
      max: 10, // attempts per window
    },
  },
});
```

### Social Providers

```typescript
// Add Google OAuth
export const auth = betterAuth({
  socialProviders: {
    google: {
      clientId: process.env.GOOGLE_CLIENT_ID!,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET!,
    },
  },
});
```

## Deployment

### Development

```bash
# Start API server
cd api && npm run dev

# Start frontend
npm run start
```

### Production

#### API Server (Vercel/Railway)

```json
// api/vercel.json
{
  "version": 2,
  "builds": [
    {
      "src": "src/app.ts",
      "use": "@vercel/node"
    }
  ],
  "routes": [
    {
      "src": "/api/(.*)",
      "dest": "src/app.ts"
    }
  ],
  "env": {
    "DATABASE_URL": "@database_url",
    "BETTER_AUTH_SECRET": "@better_auth_secret"
  }
}
```

#### Frontend (GitHub Pages)

```bash
# Build for production
npm run build

# Deploy to GitHub Pages
npm run deploy
```

## Troubleshooting

### Common Issues

1. **CORS Errors**
   - Ensure `FRONTEND_URL` is correctly set in API environment
   - Check that API server is running before starting frontend

2. **Database Connection**
   - Verify PostgreSQL is running
   - Check `DATABASE_URL` format
   - Run `npx prisma db push` to create tables

3. **Session Issues**
   - Clear browser cookies
   - Check `BETTER_AUTH_SECRET` is set
   - Verify cookie domain settings

### Environment Variables Checklist

```env
# Required for API
DATABASE_URL=postgresql://...
BETTER_AUTH_SECRET=your-256-bit-secret
FRONTEND_URL=http://localhost:3000

# Optional for email
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password

# Optional for social providers
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
```

## Next Steps

1. **Add Social Providers**: Configure Google, GitHub OAuth
2. **Email Templates**: Customize verification emails
3. **Analytics**: Add user analytics and progress tracking
4. **Mobile App**: Extend authentication to mobile applications
5. **API Documentation**: Generate comprehensive API docs

## Support

- **Documentation**: [specs/authentication/](./)
- **API Reference**: [auth-api.yaml](./contracts/auth-api.yaml)
- **Data Model**: [data-model.md](./data-model.md)
- **Issues**: Create GitHub issue for bugs or feature requests

---

**Happy coding!** 🚀

This authentication system provides a solid foundation for your Humanoid Robotics Lab Guide platform with security, scalability, and excellent user experience.