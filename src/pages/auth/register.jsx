import React from 'react';
import Layout from '@theme/Layout';
import clsx from 'clsx';
import styles from './auth.module.css';
import RegisterForm from '../../components/Auth/RegisterForm';

export default function RegisterPage() {
  return (
    <Layout
      title="Create Account"
      description="Create your Humanoid Robotics Lab Guide account"
    >
      <div className={styles.authPage}>
        <div className={styles.authContainer}>
          <div className={styles.authCard}>
            <div className={styles.authHeader}>
              <h1 className={styles.authTitle}>Join the Community</h1>
              <p className={styles.authSubtitle}>
                Create your account to start your robotics learning journey
              </p>
            </div>

            <div className={styles.authForm}>
              <RegisterForm redirectTo="/docs/intro" />
            </div>

            <div className={styles.authFooter}>
              <p className={styles.authFooterText}>
                Already have an account?{' '}
                <a href="/auth/login" className={styles.authLink}>
                  Sign in
                </a>
              </p>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
}