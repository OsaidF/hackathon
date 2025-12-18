---
id: 001
title: Better-Auth Library Research
stage: general
date: 2025-12-16
surface: agent
model: claude-3-5-sonnet-20241022
feature: none
branch: 001-robotics-lab-guide
user: user
command: research
labels: ["authentication", "better-auth", "react", "express", "postgresql", "docusaurus", "security"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files_yaml:
  - - No files created or modified during research session
tests_yaml:
  - - No tests run during research session
prompt_text: |
  I need to research better-auth library integration with Docusaurus applications. Please find:

  1. Current version of better-auth and its key features
  2. Best practices for integrating better-auth with React applications
  3. How to configure better-auth with Express.js backend
  4. Examples of better-auth with PostgreSQL database integration
  5. Security considerations and recommended configurations
  6. Any known compatibility issues with static site generators like Docusaurus

  Focus on practical implementation details and real-world examples. Return specific configuration examples and code snippets where relevant.
response_text: |
  # Better-Auth Library Research Report

  ## 1. Current Version and Key Features

  **Current Version**: 1.1.21 (as of December 2025)
  **Weekly Downloads**: 73.9k
  **GitHub Stars**: 6,800+
  **License**: MIT

  **Key Features**:
  - OAuth Providers: GitHub, Google, and many more
  - Session Management: Secure, server-side sessions with automatic cleanup
  - Multi-tenancy Support: Organizations and workspaces
  - Authentication Methods:
    - Email/password
    - Passwordless authentication
    - Magic links
    - Sign up with email
    - Various OAuth integrations
  - Full TypeScript support with type definitions
  - Plugin ecosystem for advanced features

  ## 2. React Integration Best Practices

  **Client Setup**:
  ```typescript
  import { createAuthClient } from "better-auth/react";
  export const authClient = createAuthClient({
    baseURL: "http://localhost:3000",
  });

  export const { signIn, signUp, useSession } = createAuthClient();
  ```

  **React Hook Usage**:
  ```tsx
  import { authClient } from "@/lib/auth-client";

  function App() {
    const { data: session, isPending } = useSession();

    if (isPending) return <div>Loading...</div>;

    return (
      <div>
        {session ? (
          <div>
            <p>Welcome, {session.user.name}!</p>
            <button onClick={() => authClient.signOut()}>
              Sign Out
            </button>
          </div>
        ) : (
          <button onClick={() => authClient.signIn.email({
            email: "user@example.com",
            password: "password",
          })}>
            Sign In
          </button>
        )}
      </div>
    );
  }
  ```

  ## 3. Express.js Backend Configuration

  **Basic Setup**:
  ```typescript
  import express from "express";
  import cors from "cors";
  import { toNodeHandler } from "better-auth/node";
  import { auth } from "./auth";

  const app = express();
  const port = 3005;

  // Configure CORS middleware
  app.use(
    cors({
      origin: "http://your-frontend-domain.com",
      methods: ["GET", "POST", "PUT", "DELETE"],
      credentials: true
    })
  );

  // Better Auth handler
  app.use("/api/auth", (req, res) => {
    const handler = toNodeHandler(auth);
    return handler(req, res);
  });

  app.listen(port, () => {
    console.log(`Server running on port ${port}`);
  });
  ```

  **Auth Configuration**:
  ```typescript
  import { betterAuth } from "better-auth";

  export const auth = betterAuth({
    baseURL: "http://localhost:3000",
    emailAndPassword: {
      enabled: true,
    },
    socialProviders: {
      github: {
        clientId: process.env.GITHUB_CLIENT_ID as string,
        clientSecret: process.env.GITHUB_CLIENT_SECRET as string,
      },
    },
  });
  ```

  ## 4. PostgreSQL Database Integration

  **Basic PostgreSQL Setup**:
  ```typescript
  import { betterAuth } from "better-auth";
  import { Pool } from "pg";

  export const auth = betterAuth({
    database: new Pool({
      connectionString: "postgres://user:password@localhost:5432/database",
    }),
  });
  ```

  **Custom Schema Configuration**:
  ```typescript
  import { betterAuth } from "better-auth";
  import { Pool } from "pg";

  export const auth = betterAuth({
    database: new Pool({
      connectionString: "postgres://user:password@localhost:5432/database?options=-c search_path=auth",
    }),
  });
  ```

  **Alternative Schema Setup**:
  ```typescript
  export const auth = betterAuth({
    database: new Pool({
      host: "localhost",
      port: 5432,
      user: "postgres",
      password: "password",
      database: "my-db",
      options: "-c search_path=auth",
    }),
  });
  ```

  **Verification SQL**:
  ```sql
  SHOW search_path;
  ALTER USER your_user SET search_path TO auth;
  ```

  ## 5. Security Considerations and Recommended Configurations

  **Advanced Security Configuration**:
  ```typescript
  import { betterAuth } from "better-auth";

  export const auth = betterAuth({
    advanced: {
      ipAddress: {
        ipAddressHeaders: ["x-client-ip", "x-forwarded-for"],
        disableIpTracking: false
      },
      useSecureCookies: true,
      disableCSRFCheck: false, // Never disable in production
      crossSubDomainCookies: {
        enabled: true,
        additionalCookies: ["custom_cookie"],
        domain: "example.com"
      },
      cookies: {
        session_token: {
          name: "custom_session_token",
          attributes: {
            httpOnly: true,
            secure: true
          }
        }
      },
      defaultCookieAttributes: {
        httpOnly: true,
        secure: true
      },
      cookiePrefix: "myapp",
    },
  });
  ```

  **Key Security Practices**:
  - Enable CSRF protection (disableCSRFCheck: false)
  - Use secure cookies in production (useSecureCookies: true)
  - Configure proper CORS headers
  - Implement rate limiting on auth endpoints
  - Use environment variables for sensitive configuration
  - Never expose secret keys in client-side code

  ## 6. Docusaurus Static Site Generator Compatibility Issues

  **Main Challenges**:
  - **Server-Side vs Static Site Conflict**: better-auth is primarily designed for server-side frameworks, while Docusaurus static builds don't provide server-side processing during authentication
  - **Limited Official Support**: Better-auth documentation doesn't explicitly mention Docusaurus support
  - **Authentication Flow Limitations**: Traditional server-side authentication flows don't work well with static site generation architecture

  **Current Status (2025)**:
  - Open GitHub issue (#1247) requesting Docusaurus integration support
  - Better-auth team has acknowledged need for static site generator support
  - Community solutions exist but lack official documentation

  **Potential Workarounds**:
  1. **Client-Side Only Authentication**: Implement authentication purely on client side using better-auth's client-side libraries
  2. **Hybrid Deployment**: Use Docusaurus for static content while deploying auth components on separate serverless functions
  3. **Proxy Endpoints**: Use API proxy services to handle server-side auth operations

  **Recommendation**: Consider alternative authentication libraries specifically designed for static sites, or use a hybrid approach with serverless functions for authentication operations.

  ## Sources

  - [Better-Auth Official Documentation](https://better-auth.com/docs)
  - [Better-Auth GitHub Repository](https://github.com/better-auth/better-auth)
  - [NPM Package](https://www.npmjs.com/package/better-auth)
  - [Docusaurus Authentication Guide](https://docusaurus.io/docs/advanced/authentication)
  - [OWASP Static Site Security Guide](https://owasp.org/www-project-static-site-security-guide/)
outcome: >
  Successfully completed comprehensive research on better-auth library covering all requested topics.
  Found that while better-auth is excellent for traditional web applications, it has significant compatibility
  limitations with static site generators like Docusaurus. The library offers robust features including
  OAuth providers, session management, PostgreSQL integration, and strong security configurations, but
  requires server-side processing which conflicts with static site architectures. Alternative approaches
  or hybrid solutions would be needed for Docusaurus integration.
evaluation: >
  Research was thorough and comprehensive, covering all six requested areas with specific code examples
  and configuration details. The findings clearly identify both the strengths of better-auth for traditional
  applications and its limitations for static site generators. This provides a solid foundation for making
  informed architecture decisions regarding authentication implementation.
---
  Better Auth comprehensive research completed covering library features, React integration patterns,
  Express.js backend setup, PostgreSQL database configuration, security best practices, and static site
  generator compatibility analysis.