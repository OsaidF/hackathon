import React from 'react';
import Layout from '@theme/Layout';
import clsx from 'clsx';
import styles from './auth.module.css';
import LoginForm from '../../components/Auth/LoginForm';

export default function LoginPage() {
  return (
    <Layout
      title="Sign In"
      description="Sign in to your Humanoid Robotics Lab Guide account"
    >
      <div className={styles.authPage}>
        <div className={styles.authContainer}>
          <div className={styles.authCard}>
              <LoginForm redirectTo="/docs/intro" />
          </div>
        </div>
      </div>
    </Layout>
  );
}