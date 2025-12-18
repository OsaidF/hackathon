---
id: 001
title: PostgreSQL Auth Research
stage: general
date: 2025-12-16
surface: agent
model: claude-sonnet-4-5-20250929
feature: none
branch: 001-robotics-lab-guide
user: osaid
command: research
labels: ["postgresql","authentication","database-design","performance","security"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  -
tests:
  -
---

## Prompt

I need to research PostgreSQL schema design for user authentication systems with progress tracking. Please find:

1. Best practices for user table design with UUID primary keys
2. Session management schema patterns and indexing strategies
3. Progress tracking database design for learning platforms
4. Bookmark/favoriting system schema patterns
5. Database performance optimization for authentication queries
6. Migration strategies and database versioning with Prisma
7. Security considerations for PostgreSQL authentication databases
8. Scaling considerations for 10k+ users

Focus on providing specific SQL schema examples, indexing strategies, and performance considerations for a Docusaurus-based educational platform.

## Response snapshot

## PostgreSQL Schema Design for User Authentication Systems with Progress Tracking

### 1. User Table Design with UUID Primary Keys

**Best Practices:**

```sql
-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users table with UUID primary key
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    email_verified BOOLEAN DEFAULT FALSE,
    email_verification_token UUID,
    password_reset_token UUID,
    password_reset_expires_at TIMESTAMP WITH TIME ZONE,
    last_login_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    preferences JSONB DEFAULT '{}',
    profile JSONB DEFAULT '{}',

    -- Constraints
    CONSTRAINT users_email_check CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$'),
    CONSTRAINT users_username_check CHECK (length(username) >= 3 AND username ~ '^[a-zA-Z0-9_-]+$')
);

-- Indexes for authentication queries
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email_verification_token ON users(email_verification_token) WHERE email_verification_token IS NOT NULL;
CREATE INDEX idx_users_password_reset_token ON users(password_reset_token) WHERE password_reset_token IS NOT NULL;
CREATE INDEX idx_users_last_login_at ON users(last_login_at DESC);
```

**UUID Generation Options:**
- `uuid_generate_v4()`: Random UUIDs (most common)
- `uuid_generate_v7()`: Time-ordered UUIDs (better for indexing and clustering)
- `uuid_generate_v1()`: Time-based with MAC address (reveals machine identity)

### 2. Session Management Schema Patterns

```sql
-- Sessions table for authentication tokens
CREATE TABLE user_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    refresh_token VARCHAR(255) UNIQUE,
    ip_address INET NOT NULL,
    user_agent TEXT,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_accessed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE,

    -- Device and location tracking
    device_fingerprint VARCHAR(255),
    country_code VARCHAR(2),
    city VARCHAR(100)
);

-- Indexes for session queries
CREATE INDEX idx_sessions_user_id ON user_sessions(user_id);
CREATE INDEX idx_sessions_session_token ON user_sessions(session_token);
CREATE INDEX idx_sessions_refresh_token ON user_sessions(refresh_token) WHERE refresh_token IS NOT NULL;
CREATE INDEX idx_sessions_expires_at ON user_sessions(expires_at);
CREATE INDEX idx_sessions_user_id_active ON user_sessions(user_id) WHERE is_active = TRUE;
CREATE INDEX idx_sessions_last_accessed ON user_sessions(last_accessed_at DESC);

-- Session activity logging
CREATE TABLE session_activity (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES user_sessions(id) ON DELETE CASCADE,
    activity_type VARCHAR(50) NOT NULL, -- 'login', 'logout', 'page_view', 'action'
    endpoint VARCHAR(255),
    metadata JSONB DEFAULT '{}',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Partitioned by month for better performance
CREATE INDEX idx_session_activity_session_timestamp ON session_activity(session_id, timestamp DESC);
```

### 3. Progress Tracking for Learning Platforms

```sql
-- Learning content hierarchy
CREATE TABLE courses (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
    title VARCHAR(255) NOT NULL,
    description TEXT,
    difficulty_level INTEGER CHECK (difficulty_level BETWEEN 1 AND 5),
    estimated_hours INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_published BOOLEAN DEFAULT FALSE,
    metadata JSONB DEFAULT '{}'
);

CREATE TABLE modules (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
    course_id UUID NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    order_index INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE lessons (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
    module_id UUID NOT NULL REFERENCES modules(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    content_type VARCHAR(50) NOT NULL, -- 'text', 'video', 'interactive', 'quiz'
    content JSONB NOT NULL, -- Flexible content storage
    duration_minutes INTEGER,
    order_index INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Progress tracking
CREATE TABLE user_progress (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    content_type VARCHAR(50) NOT NULL, -- 'course', 'module', 'lesson'
    content_id UUID NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'not_started', -- 'not_started', 'in_progress', 'completed'
    completion_percentage DECIMAL(5,2) DEFAULT 0.00 CHECK (completion_percentage BETWEEN 0 AND 100),
    time_sp_minutes INTEGER DEFAULT 0,
    last_accessed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}', -- Store additional progress data

    -- Ensure one progress record per user per content
    UNIQUE(user_id, content_type, content_id)
);

-- Indexes for progress queries
CREATE INDEX idx_progress_user_id ON user_progress(user_id);
CREATE INDEX idx_progress_content ON user_progress(content_type, content_id);
CREATE INDEX idx_progress_status ON user_progress(status);
CREATE INDEX idx_progress_user_status ON user_progress(user_id, status);
CREATE INDEX idx_progress_last_accessed ON user_progress(last_accessed_at DESC);
CREATE INDEX idx_progress_completion ON user_progress(status, completed_at DESC) WHERE status = 'completed';

-- Learning analytics
CREATE TABLE learning_analytics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    lesson_id UUID NOT NULL REFERENCES lessons(id) ON DELETE CASCADE,
    event_type VARCHAR(50) NOT NULL, -- 'start', 'complete', 'pause', 'seek', 'interaction'
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    duration_seconds INTEGER,
    position_seconds INTEGER, -- For video content
    metadata JSONB DEFAULT '{}'
);

-- Time-series partitioning for analytics
CREATE INDEX idx_analytics_user_lesson ON learning_analytics(user_id, lesson_id, timestamp DESC);
CREATE INDEX idx_analytics_timestamp ON learning_analytics(timestamp DESC);
```

### 4. Bookmark/Favoriting System Schema

```sql
-- Bookmarks table
CREATE TABLE bookmarks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    content_type VARCHAR(50) NOT NULL, -- 'lesson', 'module', 'course', 'resource'
    content_id UUID NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    notes TEXT,
    tags TEXT[] DEFAULT '{}',
    is_private BOOLEAN DEFAULT TRUE,
    folder_id UUID REFERENCES bookmark_folders(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(user_id, content_type, content_id)
);

-- Bookmark folders/organization
CREATE TABLE bookmark_folders (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    parent_folder_id UUID REFERENCES bookmark_folders(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(user_id, name, parent_folder_id)
);

-- Indexes for bookmark queries
CREATE INDEX idx_bookmarks_user_id ON bookmarks(user_id);
CREATE INDEX idx_bookmarks_content ON bookmarks(content_type, content_id);
CREATE INDEX idx_bookmarks_tags ON bookmarks USING GIN(tags);
CREATE INDEX idx_bookmarks_created_at ON bookmarks(created_at DESC);
CREATE INDEX idx_bookmarks_folder_id ON bookmarks(folder_id) WHERE folder_id IS NOT NULL;
CREATE INDEX idx_bookmarks_search ON bookmarks USING GIN(to_tsvector('english', title || ' ' || COALESCE(description, '') || ' ' || COALESCE(notes, '')));
```

### 5. Database Performance Optimization

**Connection Pooling Configuration (PgBouncer):**
```ini
[databases]
your_database = host=localhost port=5432 dbname=your_db

[pgbouncer]
listen_port = 6432
listen_addr = 127.0.0.1
auth_type = md5
auth_file = /etc/pgbouncer/userlist.txt
logfile = /var/log/pgbouncer/pgbouncer.log
pidfile = /var/run/pgbouncer/pgbouncer.pid
admin_users = postgres
stats_users = stats, postgres

# Pool settings for 10k+ users
max_client_conn = 10000
default_pool_size = 100
min_pool_size = 20
reserve_pool_size = 10
reserve_pool_timeout = 5
max_db_connections = 100
max_user_connections = 100

# Transaction pooling for high concurrency
pool_mode = transaction
server_reset_query = DISCARD ALL
```

**Key Indexing Strategies:**
```sql
-- B-tree indexes for equality and range queries (default, best for authentication)
CREATE INDEX CONCURRENTLY idx_users_email_active ON users(email) WHERE email_verified = TRUE;

-- Partial indexes for better performance
CREATE INDEX CONCURRENTLY idx_sessions_active ON user_sessions(user_id, expires_at) WHERE is_active = TRUE;

-- Composite indexes for common query patterns
CREATE INDEX CONCURRENTLY idx_progress_user_course ON user_progress(user_id, content_type, content_id)
WHERE content_type IN ('course', 'module');

-- GIN indexes for JSONB and array searches
CREATE INDEX CONCURRENTLY idx_users_preferences ON users USING GIN(preferences);
CREATE INDEX CONCURRENTLY idx_bookmarks_tags_gin ON bookmarks USING GIN(tags);

-- Full-text search indexes
CREATE INDEX CONCURRENTLY idx_bookmarks_search_fts ON bookmarks
USING GIN(to_tsvector('english', title || ' ' || COALESCE(description, '')));

-- BRIN indexes for time-series data (analytics)
CREATE INDEX CONCURRENTLY idx_analytics_timestamp_brin ON learning_analytics USING BRIN(timestamp);
```

**Query Optimization Examples:**
```sql
-- Efficient user authentication query
EXPLAIN (ANALYZE, BUFFERS)
SELECT u.id, u.email, u.password_hash, u.preferences
FROM users u
WHERE u.email = $1
AND u.email_verified = TRUE;

-- Optimized session lookup
EXPLAIN (ANALYZE, BUFFERS)
SELECT s.*, u.email, u.username
FROM user_sessions s
JOIN users u ON s.user_id = u.id
WHERE s.session_token = $1
AND s.expires_at > NOW()
AND s.is_active = TRUE;

-- Progress tracking with proper indexing
EXPLAIN (ANALYZE, BUFFERS)
WITH user_course_progress AS (
    SELECT
        c.id as course_id,
        c.title,
        COALESCE(SUM(up.completion_percentage) / COUNT(up.id), 0) as overall_progress,
        MAX(up.last_accessed_at) as last_activity
    FROM courses c
    LEFT JOIN modules m ON c.id = m.course_id
    LEFT JOIN lessons l ON m.id = l.module_id
    LEFT JOIN user_progress up ON (
        up.user_id = $1
        AND up.content_id IN (c.id, m.id, l.id)
        AND up.content_type = CASE
            WHEN up.content_id = c.id THEN 'course'
            WHEN up.content_id = m.id THEN 'module'
            WHEN up.content_id = l.id THEN 'lesson'
        END
    )
    WHERE c.is_published = TRUE
    GROUP BY c.id, c.title
)
SELECT * FROM user_course_progress
ORDER BY last_activity DESC NULLS LAST;
```

### 6. Migration Strategies with Prisma

**Prisma Schema Structure:**
```prisma
// prisma/schema.prisma
generator client {
  provider = "prisma-client-js"
}

datasource db {
  provider = "postgresql"
  url      = env("DATABASE_URL")
}

model User {
  id                String    @id @default(cuid())
  email             String    @unique
  username          String    @unique
  passwordHash      String    @map("password_hash")
  emailVerified     Boolean   @default(false) @map("email_verified")
  emailVerificationToken String? @unique @map("email_verification_token")
  passwordResetToken String? @unique @map("password_reset_token")
  passwordResetExpiresAt DateTime? @map("password_reset_expires_at")
  lastLoginAt       DateTime? @map("last_login_at")
  createdAt         DateTime  @default(now()) @map("created_at")
  updatedAt         DateTime  @updatedAt @map("updated_at")
  preferences       Json      @default("{}")
  profile           Json      @default("{}")

  sessions          UserSession[]
  progress          UserProgress[]
  bookmarks         Bookmark[]
  bookmarkFolders   BookmarkFolder[]

  @@map("users")
}

model UserSession {
  id              String    @id @default(cuid())
  userId          String    @map("user_id")
  sessionToken    String    @unique @map("session_token")
  refreshToken    String?   @unique @map("refresh_token")
  ipAddress       String    @map("ip_address")
  userAgent       String?   @map("user_agent")
  expiresAt       DateTime  @map("expires_at")
  createdAt       DateTime  @default(now()) @map("created_at")
  lastAccessedAt  DateTime  @default(now()) @map("last_accessed_at")
  isActive        Boolean   @default(true) @map("is_active")
  deviceFingerprint String? @map("device_fingerprint")
  countryCode     String?   @map("country_code")
  city            String?

  user            User      @relation(fields: [userId], references: [id], onDelete: Cascade)
  activities      SessionActivity[]

  @@map("user_sessions")
}

// Migration workflow scripts
// package.json
{
  "scripts": {
    "db:migrate": "npx prisma migrate dev --name",
    "db:generate": "npx prisma generate",
    "db:deploy": "npx prisma migrate deploy",
    "db:studio": "npx prisma studio",
    "db:seed": "tsx prisma/seed.ts",
    "db:reset": "npx prisma migrate reset"
  }
}
```

**Production Migration Strategy:**
```typescript
// scripts/migrate.ts
import { PrismaClient } from '@prisma/client';
import { execSync } from 'child_process';

const prisma = new PrismaClient();

async function migrate() {
  try {
    // Check if database is accessible
    await prisma.$queryRaw`SELECT 1`;

    // Deploy migrations
    execSync('npx prisma migrate deploy', { stdio: 'inherit' });

    // Generate new Prisma client
    execSync('npx prisma generate', { stdio: 'inherit' });

    console.log('Migration completed successfully');
  } catch (error) {
    console.error('Migration failed:', error);
    process.exit(1);
  } finally {
    await prisma.$disconnect();
  }
}

migrate();
```

### 7. Security Considerations

**Row-Level Security (RLS):**
```sql
-- Enable RLS on sensitive tables
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_progress ENABLE ROW LEVEL SECURITY;
ALTER TABLE bookmarks ENABLE ROW LEVEL SECURITY;

-- Users can only see their own data
CREATE POLICY user_isolation ON users
    FOR ALL
    TO authenticated_users
    USING (id = current_setting('app.current_user_id')::uuid);

CREATE POLICY user_sessions_isolation ON user_sessions
    FOR ALL
    TO authenticated_users
    USING (user_id = current_setting('app.current_user_id')::uuid);

-- Application role setup
CREATE ROLE authenticated_users;
GRANT authenticated_users TO app_user;

-- Application context setting
CREATE OR REPLACE FUNCTION set_app_context(user_id uuid)
RETURNS void AS $$
BEGIN
    PERFORM set_config('app.current_user_id', user_id::text, true);
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;
```

**Password Security:**
```sql
-- Enable pgcrypto extension
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Secure password hashing
CREATE OR REPLACE FUNCTION secure_password_hash(password text)
RETURNS text AS $$
BEGIN
    RETURN crypt(password, gen_salt('bf', 12)); -- bcrypt with cost factor 12
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Password verification
CREATE OR REPLACE FUNCTION verify_password(password text, hash text)
RETURNS boolean AS $$
BEGIN
    RETURN hash = crypt(password, hash);
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;
```

**Audit Logging:**
```sql
-- Audit table
CREATE TABLE audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID,
    table_name VARCHAR(100) NOT NULL,
    operation VARCHAR(10) NOT NULL, -- 'INSERT', 'UPDATE', 'DELETE'
    old_values JSONB,
    new_values JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ip_address INET,
    user_agent TEXT
);

-- Audit trigger function
CREATE OR REPLACE FUNCTION audit_trigger()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        INSERT INTO audit_log (user_id, table_name, operation, old_values)
        VALUES (current_setting('app.current_user_id', true)::uuid, TG_TABLE_NAME, TG_OP, row_to_json(OLD));
        RETURN OLD;
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO audit_log (user_id, table_name, operation, old_values, new_values)
        VALUES (current_setting('app.current_user_id', true)::uuid, TG_TABLE_NAME, TG_OP, row_to_json(OLD), row_to_json(NEW));
        RETURN NEW;
    ELSIF TG_OP = 'INSERT' THEN
        INSERT INTO audit_log (user_id, table_name, operation, new_values)
        VALUES (current_setting('app.current_user_id', true)::uuid, TG_TABLE_NAME, TG_OP, row_to_json(NEW));
        RETURN NEW;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Apply audit triggers
CREATE TRIGGER users_audit AFTER INSERT OR UPDATE OR DELETE ON users
    FOR EACH ROW EXECUTE FUNCTION audit_trigger();
```

### 8. Scaling Considerations for 10k+ Users

**Database Configuration Optimizations:**
```sql
-- Performance settings for postgresql.conf
-- Memory settings (adjust based on available RAM)
shared_buffers = 256MB                  -- 25% of available RAM
effective_cache_size = 1GB              -- 75% of available RAM
work_mem = 4MB                          -- Per connection
maintenance_work_mem = 64MB

-- Connection settings
max_connections = 200                   -- Limited due to connection pooling
superuser_reserved_connections = 3

-- WAL settings
wal_buffers = 16MB
checkpoint_completion_target = 0.9
wal_writer_delay = 200ms

-- Query planner settings
random_page_cost = 1.1                  -- SSD optimization
effective_io_concurrency = 200          -- SSD concurrency

-- Logging
log_min_duration_statement = 1000       -- Log slow queries (>1s)
log_checkpoints = on
log_connections = on
log_disconnections = on
```

**Partitioning Strategies:**
```sql
-- Partition user_progress by year for better performance
CREATE TABLE user_progress_partitioned (
    LIKE user_progress INCLUDING ALL
) PARTITION BY RANGE (created_at);

-- Create yearly partitions
CREATE TABLE user_progress_2025 PARTITION OF user_progress_partitioned
    FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');

CREATE TABLE user_progress_2026 PARTITION OF user_progress_partitioned
    FOR VALUES FROM ('2026-01-01') TO ('2027-01-01');

-- Partition session_activity by month (high write volume)
CREATE TABLE session_activity_partitioned (
    LIKE session_activity INCLUDING ALL
) PARTITION BY RANGE (timestamp);

-- Monthly partitions for current year
CREATE TABLE session_activity_2025_01 PARTITION OF session_activity_partitioned
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');

-- Create function to auto-create partitions
CREATE OR REPLACE FUNCTION create_monthly_partition(table_name text, start_date date)
RETURNS void AS $$
DECLARE
    partition_name text;
    end_date date;
BEGIN
    partition_name := table_name || '_' || to_char(start_date, 'YYYY_MM');
    end_date := start_date + interval '1 month';

    EXECUTE format('CREATE TABLE IF NOT EXISTS %I PARTITION OF %I
                    FOR VALUES FROM (%L) TO (%L)',
                   partition_name, table_name, start_date, end_date);
END;
$$ LANGUAGE plpgsql;
```

**Read Replicas for Scaling:**
```sql
-- Set up read replicas for analytics queries
-- Primary database handles writes and authentication
-- Read replicas handle analytics, reporting, and progress queries

-- Connection routing configuration
-- Application should route read queries to replicas:
-- - SELECT queries for analytics
-- - Progress tracking queries
-- - Bookmark and content queries
--
-- Primary database handles:
-- - Authentication (INSERT/UPDATE/DELETE users, sessions)
-- - Progress updates
-- - Real-time session management
```

**Monitoring and Maintenance:**
```sql
-- Monitoring queries for performance
SELECT
    schemaname,
    tablename,
    seq_scan,
    seq_tup_read,
    idx_scan,
    idx_tup_fetch
FROM pg_stat_user_tables
ORDER BY seq_scan DESC;

-- Index usage monitoring
SELECT
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;

-- Slow query monitoring
SELECT
    query,
    calls,
    total_time,
    mean_time,
    rows
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 10;

-- Automated maintenance
CREATE OR REPLACE FUNCTION auto_vacuum_tables()
RETURNS void AS $$
BEGIN
    -- Analyze tables with high modification rates
    PERFORM analyze_schema('public');

    -- Vacuum tables with many dead tuples
    PERFORM vacuum_table('user_sessions');
    PERFORM vacuum_table('session_activity');
    PERFORM vacuum_table('learning_analytics');
END;
$$ LANGUAGE plpgsql;

-- Schedule regular maintenance (requires pg_cron extension)
SELECT cron.schedule('auto-maintenance', '0 2 * * *', 'SELECT auto_vacuum_tables();');
```

This comprehensive schema design provides a solid foundation for a scalable, secure PostgreSQL authentication system with progress tracking suitable for a Docusaurus-based educational platform handling 10k+ users.

## Outcome

- ✅ Impact: Comprehensive PostgreSQL schema design research completed for authentication and progress tracking systems
- 🧪 Tests: N/A (research only)
- 📁 Files: Created PHR documentation for future reference
- 🔁 Next prompts: Consider implementation-specific questions or performance testing scenarios
- 🧠 Reflection: Research covered all requested areas with specific SQL examples, indexing strategies, and scaling considerations for educational platforms

## Evaluation notes (flywheel)

- Failure modes observed: Web search returned some incomplete results, but PostgreSQL documentation and Prisma docs provided comprehensive coverage
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Could explore specific Docusaurus integration patterns or create a complete Prisma schema file for immediate implementation