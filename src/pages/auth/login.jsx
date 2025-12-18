import React from 'react';
import Layout from '@theme/Layout';
import clsx from 'clsx';
import styles from './auth.module.css';
import LoginForm from '../../components/Auth/LoginForm.tsx';

export default function LoginPage() {
  return (
    <Layout
      title="Sign In"
      description="Sign in to your Humanoid Robotics Lab Guide account"
    >
      <div className={styles.authPage}>
        <div className={styles.authContainer}>
          <div className={styles.authCard}>
            <div className={styles.authHeader}>
              <h1 className={styles.authTitle}>Welcome Back</h1>
              <p className={styles.authSubtitle}>
                Sign in to access your personalized learning journey
              </p>
            </div>

            <div className={styles.authForm}>
              <LoginForm redirectTo="/docs/intro" />
            </div>

            <div className={styles.authFooter}>
              <p className={styles.authFooterText}>
                Don't have an account?{' '}
                <a href="/auth/register" className={styles.authLink}>
                  Sign up
                </a>
              </p>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
}