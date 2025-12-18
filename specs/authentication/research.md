# Authentication Research Findings

**Date**: 2024-12-18
**Feature**: Authentication for Humanoid Robotics Lab Guide
**Research Focus**: better-auth + Express.js + Docusaurus integration

## Executive Summary

Research indicates that better-auth with default configuration provides a solid foundation for authentication in our Docusaurus documentation site. The separate API server approach with custom Docusaurus plugin integration offers the best balance of security, performance, and maintainability while preserving GitHub Pages deployment compatibility.

## Key Decisions

### 1. Authentication Backend Technology

**Decision**: Use better-auth with default configuration and PostgreSQL adapter

**Rationale**:
- Default better-auth configuration follows security best practices out-of-the-box
- PostgreSQL adapter provides excellent performance and reliability
- Built-in session management reduces custom code complexity
- Strong community support and active development

**Alternatives Considered**:
- Custom JWT implementation (higher complexity, more security surface area)
- Auth0/Firebase (vendor lock-in, costs for production)
- NextAuth.js (better aligned with Next.js, less optimal for Docusaurus)

### 2. Frontend Integration Pattern

**Decision**: Custom Docusaurus plugin with React Context for auth state management

**Rationale**:
- Seamless integration with existing Docusaurus architecture
- React Context provides centralized state management
- Plugin approach allows for clean separation of concerns
- Maintains static build compatibility for GitHub Pages

**Alternatives Considered**:
- Standalone React SPA (complex routing, poor UX)
- Server-side auth (incompatible with static hosting)
- Client-side only (limited functionality)

### 3. Form Handling Strategy

**Decision**: React Hook Form + Zod for form validation

**Rationale**:
- Excellent TypeScript support with type safety
- Superior performance with minimal re-renders
- Zod provides schema-first validation
- Strong developer experience with clear error handling

**Alternatives Considered**:
- Formik + Yup (more verbose, performance overhead)
- Controlled components (manual state management, more code)
- HTML5 forms only (limited validation, poor UX)

## Technical Implementation Details

### API Server Configuration

```javascript
// auth.config.js
import { betterAuth } from "better-auth";
import { postgresAdapter } from "@better-auth/postgres";
import { Pool } from "pg";

const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  max: 20,
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 2000,
  ssl: process.env.NODE_ENV === 'production' ? { rejectUnauthorized: false } : false
});

export const auth = betterAuth({
  database: postgresAdapter(pool),
  emailAndPassword: {
    enabled: true,
    requireEmailVerification: true,
    minPasswordLength: 8,
    maxPasswordLength: 128
  },
  session: {
    expiresIn: 60 * 60 * 24 * 7, // 7 days
    updateAge: 60 * 60 * 24, // 1 day
    cookieCache: {
      enabled: true,
      maxAge: 5 * 60 // 5 minutes
    }
  },
  security: {
    csrfProtection: {
      enabled: true,
      trustedOrigins: [process.env.FRONTEND_URL]
    },
    rateLimit: {
      enabled: true,
      window: 60 * 1000, // 1 minute
      max: 10 // attempts per window
    }
  }
});
```

### Database Schema Optimization

**Decision**: Use better-auth's default PostgreSQL schema with additional indexes

```sql
-- Performance indexes for better-auth tables
CREATE INDEX CONCURRENTLY idx_user_email ON users(email);
CREATE INDEX CONCURRENTLY idx_session_user_id ON sessions(user_id);
CREATE INDEX CONCURRENTLY idx_session_token ON sessions(token);
CREATE INDEX CONCURRENTLY idx_session_expires ON sessions(expires);

-- Custom tables for progress tracking
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

CREATE INDEX CONCURRENTLY idx_user_progress_user_id ON user_progress(user_id);
CREATE INDEX CONCURRENTLY idx_user_progress_content_path ON user_progress(content_path);
```

### CORS Configuration

**Critical Requirement**: Proper CORS setup for separate frontend/backend deployment

```javascript
// cors.config.js
const corsOptions = {
  origin: (origin, callback) => {
    const allowedOrigins = [
      process.env.FRONTEND_URL,
      'http://localhost:3000',
      'http://localhost:3001',
      'https://docs.humanoid-robotics.com'
    ];

    if (!origin) return callback(null, true);

    if (allowedOrigins.includes(origin)) {
      callback(null, true);
    } else {
      callback(new Error('Not allowed by CORS'));
    }
  },
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: [
    'Content-Type',
    'Authorization',
    'X-Requested-With',
    'X-CSRF-Token'
  ],
  maxAge: 86400 // 24 hours
};
```

## Security Considerations

### Session Management
- **HTTP-only cookies**: Prevent XSS attacks
- **Secure flags**: HTTPS-only in production
- **CSRF protection**: Built-in better-auth protection
- **SameSite cookies**: Proper 'lax' setting for UX/security balance

### Rate Limiting
- **Authentication endpoints**: 5 attempts per 15 minutes
- **General API**: 100 requests per 15 minutes
- **Password reset**: 3 attempts per hour

### Password Security
- **Minimum length**: 8 characters
- **Hashing**: bcrypt with salt rounds (handled by better-auth)
- **Reset tokens**: Single-use, time-limited

## Performance Optimization

### Database Connection Pooling
```javascript
const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  max: 20, // Maximum connections
  min: 2,  // Minimum connections
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 2000,
});
```

### Caching Strategy
- **Session cache**: 5-minute in-memory cache
- **User data**: Optional Redis integration for production scaling
- **Static assets**: Continue using GitHub Pages CDN

### Frontend Performance
- **Lazy loading**: Auth components loaded on-demand
- **Bundle splitting**: Separate auth bundle from main documentation
- **Smart polling**: Only check auth status when necessary

## Deployment Architecture

### Development Environment
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Docusaurus      │    │ Express.js       │    │ PostgreSQL      │
│ (localhost:3000)│◄──►│ (localhost:4000) │◄──►│ (localhost:5432) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Production Environment
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ GitHub Pages    │    │ Railway/Vercel   │    │ Supabase/Neon   │
│ (Static Site)   │◄──►│ (API Server)     │◄──►│ (Managed PG)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Risk Assessment

### High Risk
- **CORS configuration**: Must be properly configured for production
- **Session security**: Requires proper HTTPS and secure cookie configuration
- **Database performance**: Connection pooling essential for scaling

### Medium Risk
- **Rate limiting**: Must prevent abuse while not blocking legitimate users
- **Email delivery**: Password reset functionality requires reliable email service
- **User experience**: Progressive enhancement must work gracefully

### Low Risk
- **Form validation**: Well-established patterns with React Hook Form + Zod
- **Database schema**: better-auth provides proven schema patterns
- **Frontend integration**: Docusaurus plugin architecture is well-documented

## Implementation Timeline

### Phase 1: Infrastructure (Days 1-3)
- Set up Express.js API server
- Configure better-auth with PostgreSQL
- Implement basic auth endpoints
- Set up CORS and security middleware

### Phase 2: Frontend Integration (Days 4-6)
- Create custom Docusaurus plugin
- Implement React Context for auth state
- Build auth components with React Hook Form + Zod
- Set up protected routes

### Phase 3: UI/UX Enhancement (Days 7-9)
- Implement auth-aware navbar
- Add toast notifications
- Create user dashboard
- Add progress tracking

### Phase 4: Testing & Deployment (Days 10-12)
- Comprehensive testing (unit, integration, e2e)
- Security audit
- Production deployment
- Performance optimization

## Success Metrics

### Technical Metrics
- **Authentication success rate**: >95%
- **API response time**: <200ms average
- **Database query time**: <50ms average
- **Uptime**: >99.9%

### User Experience Metrics
- **Registration conversion**: Target 10-15%
- **Login completion rate**: >90%
- **Error rate**: <1% for authentication flows
- **Mobile compatibility**: 100% feature parity

## Sources

1. [Better Auth Official Documentation](https://better-auth.com/docs)
2. [Better Auth PostgreSQL Adapter Guide](https://better-auth.com/docs/adapters/postgresql)
3. [Docusaurus Plugin Development Guide](https://docusaurus.io/docs/advanced/plugins)
4. [React Hook Form Documentation](https://react-hook-form.com/)
5. [Zod Validation Library](https://zod.dev/)
6. [OWASP Authentication Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html)
7. [CORS Best Practices 2025](https://web.dev/cors-best-practices-2025)

---

**Next Steps**: Proceed to Phase 1 design to create data models, API contracts, and quickstart documentation based on these research findings.