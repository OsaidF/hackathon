-- Database Schema for Humanoid Robotics Lab Guide Authentication System
-- Version: 1.0
-- Created: 2025-12-16
-- Database: PostgreSQL 14+

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create schema for better organization (optional)
-- CREATE SCHEMA IF NOT EXISTS auth;
-- SET search_path TO auth, public;

-- =============================================
-- TABLE DEFINITIONS
-- =============================================

-- Users table - synchronized with Auth0
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
    auth0_id VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255),
    avatar_url VARCHAR(500),
    email_verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Constraints
    CONSTRAINT users_email_check CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$'),
    CONSTRAINT users_auth0_id_check CHECK (auth0_id ~* '^auth0\|[0-9a-f]{24}$')
);

-- User progress tracking table
CREATE TABLE user_progress (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    content_path VARCHAR(255) NOT NULL,
    completion_percentage INTEGER DEFAULT 0 CHECK (completion_percentage >= 0 AND completion_percentage <= 100),
    time_spent_minutes INTEGER DEFAULT 0 CHECK (time_spent_minutes >= 0),
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, content_path)
);

-- Bookmarks table
CREATE TABLE bookmarks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    content_path VARCHAR(255) NOT NULL,
    title VARCHAR(255),
    notes TEXT,
    folder VARCHAR(100) DEFAULT 'default',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, content_path)
);

-- User sessions table (for enhanced tracking)
CREATE TABLE user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User preferences table
CREATE TABLE user_preferences (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v7(),
    user_id UUID UNIQUE NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    preferences JSONB DEFAULT '{}',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =============================================
-- INDEXES
-- =============================================

-- Users table indexes
CREATE INDEX idx_users_auth0_id ON users(auth0_id);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_created_at ON users(created_at);

-- User progress indexes
CREATE INDEX idx_progress_user_content ON user_progress(user_id, content_path);
CREATE INDEX idx_progress_user_last ON user_progress(user_id, last_accessed DESC);
CREATE INDEX idx_progress_content_path ON user_progress(content_path);
CREATE INDEX idx_progress_completion ON user_progress(completion_percentage) WHERE completion_percentage = 100;

-- Bookmarks indexes
CREATE INDEX idx_bookmarks_user_folder ON bookmarks(user_id, folder);
CREATE INDEX idx_bookmarks_user_created ON bookmarks(user_id, created_at DESC);
CREATE INDEX idx_bookmarks_content_path ON bookmarks(content_path);

-- Full-text search index for bookmarks
CREATE INDEX idx_bookmarks_search ON bookmarks USING gin(to_tsvector('english', title || ' ' || COALESCE(notes, '')));

-- User sessions indexes
CREATE INDEX idx_sessions_user_id ON user_sessions(user_id);
CREATE INDEX idx_sessions_token ON user_sessions(session_token);
CREATE INDEX idx_sessions_expires ON user_sessions(expires_at);

-- User preferences indexes
CREATE INDEX idx_preferences_user_id ON user_preferences(user_id);

-- =============================================
-- TRIGGERS AND FUNCTIONS
-- =============================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updated_at columns
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_progress_updated_at
    BEFORE UPDATE ON user_progress
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_preferences_updated_at
    BEFORE UPDATE ON user_preferences
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to clean up expired sessions
CREATE OR REPLACE FUNCTION cleanup_expired_sessions()
RETURNS void AS $$
BEGIN
    DELETE FROM user_sessions WHERE expires_at < CURRENT_TIMESTAMP;
END;
$$ LANGUAGE plpgsql;

-- =============================================
-- VIEWS
-- =============================================

-- User progress summary view
CREATE VIEW user_progress_summary AS
SELECT
    u.id as user_id,
    u.email,
    COUNT(up.id) as total_content_items,
    SUM(up.completion_percentage) as total_completion,
    ROUND(AVG(up.completion_percentage), 2) as avg_completion,
    SUM(up.time_spent_minutes) as total_time_spent,
    COUNT(CASE WHEN up.completion_percentage = 100 THEN 1 END) as completed_items,
    MAX(up.last_accessed) as last_activity
FROM users u
LEFT JOIN user_progress up ON u.id = up.user_id
GROUP BY u.id, u.email;

-- User statistics view
CREATE VIEW user_statistics AS
SELECT
    u.id,
    u.email,
    u.created_at as user_since,
    COALESCE(ps.total_content_items, 0) as content_items_tracked,
    COALESCE(ps.completed_items, 0) as completed_items,
    COALESCE(ps.avg_completion, 0) as overall_completion,
    COALESCE(bm.bookmark_count, 0) as total_bookmarks,
    COALESCE(ps.last_activity, u.created_at) as last_activity
FROM users u
LEFT JOIN user_progress_summary ps ON u.id = ps.user_id
LEFT JOIN (
    SELECT user_id, COUNT(*) as bookmark_count
    FROM bookmarks
    GROUP BY user_id
) bm ON u.id = bm.user_id;

-- =============================================
-- SECURITY AND ACCESS CONTROL
-- =============================================

-- Row Level Security (RLS) - Enable if needed
-- ALTER TABLE users ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE user_progress ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE bookmarks ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE user_preferences ENABLE ROW LEVEL SECURITY;

-- RLS Policies (example - customize based on requirements)
/*
-- Users can only see their own data
CREATE POLICY user_progress_policy ON user_progress
    FOR ALL TO app_user
    USING (user_id = current_setting('app.current_user_id')::UUID);

CREATE POLICY bookmarks_policy ON bookmarks
    FOR ALL TO app_user
    USING (user_id = current_setting('app.current_user_id')::UUID);

CREATE POLICY user_preferences_policy ON user_preferences
    FOR ALL TO app_user
    USING (user_id = current_setting('app.current_user_id')::UUID);
*/

-- =============================================
-- SAMPLE DATA (for development)
-- =============================================

-- Sample user (replace with actual Auth0 user ID)
INSERT INTO users (auth0_id, email, name, email_verified) VALUES
('auth0|615c8f3d8b3d6b001f8e4d5e', 'john.doe@example.com', 'John Doe', true),
('auth0|615c8f3d8b3d6b001f8e4d5f', 'jane.smith@example.com', 'Jane Smith', true);

-- Sample progress data
INSERT INTO user_progress (user_id, content_path, completion_percentage, time_spent_minutes) VALUES
((SELECT id FROM users WHERE email = 'john.doe@example.com'), '/docs/quarter-1/ros2-fundamentals', 75, 45),
((SELECT id FROM users WHERE email = 'john.doe@example.com'), '/docs/quarter-1/communication-patterns', 100, 30),
((SELECT id FROM users WHERE email = 'jane.smith@example.com'), '/docs/quarter-2/simulation-basics', 50, 25);

-- Sample bookmarks
INSERT INTO bookmarks (user_id, content_path, title, notes, folder) VALUES
((SELECT id FROM users WHERE email = 'john.doe@example.com'), '/docs/quarter-1/ros2-fundamentals', 'ROS2 Fundamentals', 'Important foundation concepts', 'quarter-1'),
((SELECT id FROM users WHERE email = 'jane.smith@example.com'), '/docs/quarter-2/simulation-basics', 'Simulation Basics', 'Need to review physics simulation', 'to-review');

-- Sample preferences
INSERT INTO user_preferences (user_id, preferences) VALUES
((SELECT id FROM users WHERE email = 'john.doe@example.com'), '{"theme": "dark", "language": "en", "notifications": {"email": true, "progress_updates": true}}'::jsonb),
((SELECT id FROM users WHERE email = 'jane.smith@example.com'), '{"theme": "light", "language": "en", "notifications": {"email": false, "progress_updates": true}}'::jsonb);

-- =============================================
-- MAINTENANCE AND MONITORING
-- =============================================

-- Scheduled cleanup (requires pg_cron extension)
-- SELECT cron.schedule('cleanup-expired-sessions', '0 */6 * * *', 'SELECT cleanup_expired_sessions();');

-- Monitoring queries
-- User growth over time
/*
SELECT
    DATE_TRUNC('month', created_at) as month,
    COUNT(*) as new_users
FROM users
GROUP BY month
ORDER BY month DESC;
*/

-- Most popular content
/*
SELECT
    content_path,
    COUNT(*) as user_count,
    AVG(completion_percentage) as avg_completion
FROM user_progress
GROUP BY content_path
ORDER BY user_count DESC
LIMIT 10;
*/

-- User engagement metrics
/*
SELECT
    COUNT(DISTINCT user_id) as active_users,
    AVG(completion_percentage) as avg_progress,
    SUM(time_spent_minutes) as total_time
FROM user_progress
WHERE last_accessed > CURRENT_DATE - INTERVAL '30 days';
*/

-- =============================================
-- BACKUP AND RECOVERY CONSIDERATIONS
-- =============================================

-- Important tables to backup:
-- - users (user identity data)
-- - user_progress (learning analytics)
-- - bookmarks (user-saved content)
-- - user_preferences (user settings)

-- Backup strategy:
-- 1. Daily full backups
-- 2. Hourly transaction log backups
-- 3. Point-in-time recovery capability (7-day retention)

-- Recovery testing:
-- -- Test restore procedures monthly
-- -- Validate data integrity after restore
-- -- Document recovery time objectives (RTO) and recovery point objectives (RPO)

-- =============================================
-- PERFORMANCE TUNING NOTES
-- =============================================

-- Connection pooling configuration (PgBouncer):
-- - Pool size: 20-50 connections per application instance
-- - Pool mode: transaction
-- - Server reset query: DISCARD ALL

-- Memory settings for PostgreSQL:
-- - shared_buffers: 25% of system memory
-- - effective_cache_size: 75% of system memory
-- - work_mem: 4MB per connection
-- - maintenance_work_mem: 64MB

-- Autovacuum settings:
-- - autovacuum = on
-- - autovacuum_max_workers = 3
-- - autovacuum_naptime = 1min

-- Query optimization:
-- - Use EXPLAIN ANALYZE for slow queries
-- - Monitor with pg_stat_statements
-- - Create composite indexes for complex queries
-- - Consider partitioning for large tables (>10M rows)

-- =============================================
-- SECURITY NOTES
-- =============================================

-- Database security best practices:
-- 1. Use SSL/TLS for all connections
-- 2. Implement least privilege access
-- 3. Regular security updates
-- 4. Audit logging for sensitive operations
-- 5. Encrypted backups

-- Application security considerations:
-- 1. Parameterized queries (prevent SQL injection)
-- 2. Input validation and sanitization
-- 3. Rate limiting for API endpoints
-- 4. Proper error handling (don't expose sensitive information)
-- 5. Regular security audits and penetration testing