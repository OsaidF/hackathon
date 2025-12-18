# Authentication Feature Specification

## Overview
Add user authentication to the Humanoid Robotics Lab Guide platform using better-auth with PostgreSQL database. This will enhance the current Docusaurus documentation site with user accounts, progress tracking, and personalized learning experiences.

## Clarifications

### Session 2024-12-18
- Q: What frontend integration approach should be used? → A: Docusaurus plugin integration - Custom plugin with React Context for auth state
- Q: Which form handling library should be used? → A: React Hook Form + Zod - Modern form handling with validation
- Q: How should auth UI be integrated into the site? → A: Auth-aware navbar - Integrate user menu into existing navigation
- Q: What content should require authentication? → A: Private pages + dashboard - Protect user areas, docs stay public
- Q: How should auth feedback be displayed to users? → A: Toast + inline errors - Toast for success, inline for validation

## Current State Analysis
- **Platform Type**: Docusaurus 3.9.2 static documentation site
- **Tech Stack**: React 18.2.0, Docusaurus 3.9.2, TypeScript
- **Current Authentication**: None (public site)
- **Database**: None (static content)
- **Deployment**: GitHub Pages (static hosting)

## Target Architecture

### Docusaurus-Native Approach
**Plugin-Based Architecture**: Use Docusaurus plugins and custom client-side routing with a separate Express.js API server for authentication.

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Docusaurus    │    │   Express.js     │    │   PostgreSQL    │
│   Frontend      │◄──►│   API Server     │◄──►│   Database      │
│   (React SPA)   │    │   /api/auth/*    │    │   (Users,       │
│   Client Routes │    │   better-auth    │    │    Sessions)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Authentication Features

### 1. Core Authentication
- **Email/Password Authentication**
- **Password Reset via Email**
- **Email Verification**
- **Session Management**
- **Social Logins** (Google, GitHub - optional)

### 2. User Management
- **User Registration**
- **Profile Management**
- **Account Settings**
- **Password Change**
- **Account Deletion**

### 3. Access Control
- **Public Documentation** (no auth required)
- **User Progress Tracking** (auth required)
- **Bookmarking Content** (auth required)
- **Personalized Dashboard** (auth required)

### 4. Session Handling
- **Secure HTTP-only Cookies**
- **JWT Tokens**
- **Session Persistence**
- **Automatic Logout**
- **Cross-Site Request Forgery (CSRF) Protection**

## Technical Implementation

### Database Schema

```sql
-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    email_verified BOOLEAN DEFAULT FALSE,
    name VARCHAR(255),
    password_hash VARCHAR(255),
    image VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sessions table
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token VARCHAR(255) UNIQUE NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Accounts table (for social providers)
CREATE TABLE accounts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    provider VARCHAR(255) NOT NULL,
    provider_account_id VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(provider, provider_account_id)
);

-- User progress tracking
CREATE TABLE user_progress (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    content_path VARCHAR(255) NOT NULL,
    completed BOOLEAN DEFAULT FALSE,
    completion_percentage INTEGER DEFAULT 0,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, content_path)
);

-- Bookmarks
CREATE TABLE bookmarks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    content_path VARCHAR(255) NOT NULL,
    title VARCHAR(255),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, content_path)
);
```

### Technology Stack

**API Server (Separate from Docusaurus):**
- **Express.js** (Node.js backend)
- **better-auth** (latest version)
- **PostgreSQL** (database)
- **Prisma** (ORM)
- **CORS** (for Docusaurus frontend)
- **TypeScript**

**Docusaurus Integration:**
- **Custom Docusaurus Plugin** for auth with React Context
- **React Hook Form + Zod** for form validation
- **Auth-aware navbar** with user menu integration
- **Toast notifications** (react-hot-toast) for success messages
- **Inline error display** for form validation
- **Protected routes** for user-specific pages only

### Environment Variables

```env
# Database
DATABASE_URL="postgresql://username:password@localhost:5432/humanoid_robotics"

# Better Auth (API Server)
BETTER_AUTH_SECRET="your-secret-key"
BETTER_AUTH_URL="http://localhost:3001"

# Email (for password reset)
SMTP_HOST="smtp.gmail.com"
SMTP_PORT=587
SMTP_USER="your-email@gmail.com"
SMTP_PASSWORD="your-app-password"

# OAuth (optional)
GOOGLE_CLIENT_ID="your-google-client-id"
GOOGLE_CLIENT_SECRET="your-google-client-secret"
GITHUB_CLIENT_ID="your-github-client-id"
GITHUB_CLIENT_SECRET="your-github-client-secret"

# Frontend (Docusaurus)
REACT_APP_API_URL="http://localhost:3001"
```

## Implementation Plan

### Phase 1: API Server Setup
1. **Express.js API Server**
   - Initialize Node.js project
   - Set up Express server
   - Configure better-auth
   - Create auth endpoints

2. **Database Setup**
   - Install PostgreSQL
   - Create database schema
   - Set up Prisma

### Phase 2: Docusaurus Integration
1. **Custom Docusaurus Plugin**
   - Create plugin structure
   - Configure auth client
   - Add auth pages to routes

2. **React Components**
   - Auth context provider
   - Login/Register components
   - Protected route wrapper

### Phase 3: UI/UX Enhancement
1. **Theme Customization**
   - Auth-aware navigation
   - User menu/profile
   - Progress indicators

2. **Progress Tracking**
   - Content completion tracking
   - Bookmark functionality
   - Personalized dashboard

## File Structure Changes

```
d:\Hackathon\humanoid-robotics\
├── api/                    # New Express.js API server
│   ├── src/
│   │   ├── app.ts          # Express app setup
│   │   ├── routes/
│   │   │   └── auth.ts     # Auth routes
│   │   ├── lib/
│   │   │   ├── auth.ts     # better-auth configuration
│   │   │   └── db.ts       # database connection
│   │   └── middleware/
│   │       └── cors.ts     # CORS setup
│   ├── prisma/
│   │   ├── schema.prisma   # database schema
│   │   └── migrations/
│   ├── package.json        # API server dependencies
│   └── tsconfig.json       # TypeScript config
├── src/
│   ├── components/
│   │   ├── Auth/           # New auth components
│   │   │   ├── LoginForm.tsx
│   │   │   ├── RegisterForm.tsx
│   │   │   ├── ProfileSettings.tsx
│   │   │   ├── ProtectedRoute.tsx
│   │   │   └── AuthProvider.tsx
│   │   ├── Progress/       # Progress tracking
│   │   │   ├── ProgressBar.tsx
│   │   │   ├── ProgressTracker.tsx
│   │   │   └── BookmarkButton.tsx
│   │   └── UserMenu/       # User navigation
│   │       ├── UserMenu.tsx
│   │       └── UserProfile.tsx
│   ├── pages/
│   │   ├── auth/
│   │   │   ├── login.jsx
│   │   │   ├── register.jsx
│   │   │   ├── reset-password.jsx
│   │   │   └── profile.jsx
│   │   └── dashboard.jsx   # User dashboard
│   ├── theme/
│   │   ├── NavbarItem/     # Auth-aware navigation
│   │   └── Layout/
│   │       └── AuthLayout.tsx
│   ├── lib/
│   │   ├── auth.ts         # Auth client configuration
│   │   └── api.ts          # API client utilities
│   └── plugins/
│       └── auth-plugin.js  # Custom Docusaurus plugin
├── docusaurus.config.js    # Updated with auth plugin
├── .env.local             # Environment variables
└── package.json           # Updated dependencies
```

## Dependencies to Add

### API Server Dependencies
```json
{
  "dependencies": {
    "express": "^4.18.2",
    "better-auth": "^0.8.0",
    "@prisma/client": "^5.0.0",
    "prisma": "^5.0.0",
    "bcryptjs": "^2.4.3",
    "cors": "^2.8.5",
    "helmet": "^7.0.0",
    "jsonwebtoken": "^9.0.0",
    "nodemailer": "^6.9.0",
    "express-rate-limit": "^6.10.0"
  },
  "devDependencies": {
    "@types/express": "^4.17.21",
    "@types/bcryptjs": "^2.4.6",
    "@types/jsonwebtoken": "^9.0.5",
    "@types/nodemailer": "^6.4.14",
    "@types/cors": "^2.8.17",
    "typescript": "^5.0.0",
    "ts-node": "^10.9.0"
  }
}
```

### Docusaurus Dependencies
```json
{
  "dependencies": {
    "better-auth": "^0.8.0",
    "react-hook-form": "^7.48.0",
    "zod": "^3.22.0",
    "@hookform/resolvers": "^3.3.0",
    "react-hot-toast": "^2.4.0",
    "axios": "^1.6.0",
    "react-router-dom": "^6.8.0"
  },
  "devDependencies": {
    "@types/react-router-dom": "^5.3.3"
  }
}
```

## Docusaurus Plugin Configuration

### Custom Auth Plugin (`src/plugins/auth-plugin.js`)
```javascript
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

### Updated `docusaurus.config.js`
```javascript
const config = {
  // ... existing config
  plugins: [
    // ... existing plugins
    [require.resolve('./src/plugins/auth-plugin.js'), {
      apiUrl: process.env.REACT_APP_API_URL || 'http://localhost:3001',
    }],
  ],
  themeConfig: {
    navbar: {
      items: [
        // ... existing items
        {
          type: 'custom-auth',
          position: 'right',
        },
      ],
    },
  },
};
```

## Client-Side Routing Setup

### Auth Provider (`src/components/Auth/AuthProvider.tsx`)
```typescript
import React, { createContext, useContext, useState, useEffect } from 'react';
import { createAuthClient } from '@auth/client';

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

  const authClient = createAuthClient({
    baseURL: process.env.REACT_APP_API_URL || 'http://localhost:3001',
  });

  useEffect(() => {
    checkAuth();
  }, []);

  const checkAuth = async () => {
    try {
      const session = await authClient.getSession();
      setUser(session.user);
    } catch (error) {
      setUser(null);
    } finally {
      setLoading(false);
    }
  };

  const login = async (email: string, password: string) => {
    // Login implementation
  };

  const register = async (email: string, password: string, name: string) => {
    // Register implementation
  };

  const logout = async () => {
    await authClient.signOut();
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ user, login, register, logout, loading }}>
      {children}
    </AuthContext.Provider>
  );
};
```

## API Server Implementation

### Express App Setup (`api/src/app.ts`)
```typescript
import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';
import { betterAuth } from 'better-auth';
import { authRoutes } from './routes/auth';

const app = express();

// Security middleware
app.use(helmet());
app.use(cors({
  origin: process.env.FRONTEND_URL || 'http://localhost:3000',
  credentials: true,
}));

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // limit each IP to 100 requests per windowMs
});
app.use(limiter);

// Body parsing
app.use(express.json());

// Auth routes
app.use('/api/auth', authRoutes);

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'ok' });
});

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
  console.log(`Auth server running on port ${PORT}`);
});

export default app;
```

## Security Considerations

### 1. API Security
- **CORS Configuration**: Restrict to Docusaurus domain
- **Rate Limiting**: Prevent brute force attacks
- **Helmet.js**: Security headers
- **Input Validation**: Sanitize all inputs

### 2. Session Security
- **HTTP-only Cookies**: Prevent XSS attacks
- **Secure Flags**: HTTPS-only in production
- **CSRF Protection**: Double submit cookies or tokens
- **SameSite Cookies**: Proper cookie settings

### 3. Database Security
- **Parameterized Queries**: Via Prisma ORM
- **Connection Security**: SSL/TLS for database connections
- **Access Control**: Least privilege principle
- **Regular Backups**: Automated database backups

## Performance Considerations

### 1. Database Optimization
- **Indexes**: On email, session tokens, user_id fields
- **Connection Pooling**: Efficient database connections
- **Query Optimization**: Efficient queries with proper selects

### 2. Frontend Performance
- **Lazy Loading**: Auth components loaded on-demand
- **State Management**: Efficient auth state updates
- **Bundle Splitting**: Separate auth bundle from main bundle
- **Caching**: API response caching where appropriate

## Deployment Strategy

### 1. Development
- **Local Development**: Two servers (Docusaurus + API)
- **Database**: Local PostgreSQL or Docker
- **Environment Variables**: `.env.local`

### 2. Production
- **API Server**: Vercel, Railway, or DigitalOcean
- **Database**: Managed PostgreSQL (Supabase, Neon, PlanetScale)
- **Frontend**: GitHub Pages (existing) + API endpoints
- **Environment**: Production environment variables
- **CDN**: For static assets

### Deployment Options
1. **Monorepo**: Single repo with both frontend and API
2. **Separate Repos**: Frontend and API in different repositories
3. **Serverless**: API deployed as serverless functions

## Migration Strategy

### 1. Gradual Rollout
- **Feature Flags**: Enable authentication progressively
- **Beta Testing**: Test with subset of users
- **Fallback**: Public content remains accessible

### 2. Content Migration
- **Existing Content**: No migration needed (static docs)
- **User Data**: Fresh start with empty database
- **SEO Impact**: Maintain existing URLs and content structure

## Success Metrics

### 1. Technical Metrics
- **Authentication Success Rate**: >95%
- **API Response Time**: <200ms average
- **Database Query Time**: <50ms average
- **Uptime**: >99.9%

### 2. User Metrics
- **Registration Conversion Rate**: Target 10-15%
- **User Engagement**: Track progress completion rates
- **Support Tickets**: Monitor auth-related issues
- **Session Duration**: Measure user engagement

## Risk Assessment

### High Risk
- **Separate Servers**: Managing two deployments
- **CORS Issues**: Cross-origin request problems
- **State Synchronization**: Keeping auth state in sync

### Medium Risk
- **Performance**: Additional network requests
- **Security**: Implementing auth correctly
- **User Experience**: Smooth integration with existing UI

### Low Risk
- **Dependency Management**: Adding new packages
- **Development Complexity**: Learning better-auth
- **Testing**: Comprehensive test coverage

## Rollback Plan

### 1. Immediate Rollback
- **API Server**: Deploy previous version
- **Frontend**: Disable auth plugin temporarily
- **Database**: Backup before schema changes

### 2. Graceful Degradation
- **Authentication**: Disable auth features, keep site public
- **API Routes**: Return static responses
- **Database**: Temporary unavailability handling

## Testing Strategy

### 1. Unit Tests
- **Auth Functions**: Login, register, logout
- **API Endpoints**: All auth endpoints
- **Database Operations**: User creation, session management

### 2. Integration Tests
- **Frontend + API**: End-to-end auth flows
- **Database + API**: Data persistence
- **Cross-browser**: Compatibility testing

### 3. Security Tests
- **Authentication**: Password strength, session security
- **Authorization**: Access control testing
- **Vulnerability Scanning**: Security audit

## Next Steps

1. **Setup API Server**: Create Express.js project structure
2. **Configure Database**: Install PostgreSQL and create schema
3. **Implement better-auth**: Set up authentication endpoints
4. **Create Docusaurus Plugin**: Integrate auth with Docusaurus
5. **Build UI Components**: Login/register forms, user menu
6. **Test**: Comprehensive testing of authentication flow
7. **Deploy**: Production deployment with proper monitoring

---

## Acceptance Criteria

- [ ] Users can register with email/password
- [ ] Users can log in and log out securely
- [ ] Password reset functionality works
- [ ] Email verification is implemented
- [ ] User progress is tracked in database
- [ ] Authentication state persists across page refreshes
- [ ] Public documentation remains accessible without auth
- [ ] Bookmarking functionality works for authenticated users
- [ ] User dashboard displays progress and bookmarks
- [ ] Security best practices are implemented
- [ ] Performance impact is minimal (<100ms additional API call time)
- [ ] All authentication features work on mobile devices
- [ ] Error handling provides clear user feedback
- [ ] CORS is properly configured between API and frontend
- [ ] Database schema is properly indexed and optimized

---

*This specification provides a Docusaurus-native approach to implementing authentication while maintaining the existing documentation structure and leveraging the React-based architecture of Docusaurus.*