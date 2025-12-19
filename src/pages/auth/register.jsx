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
              <RegisterForm redirectTo="/docs/intro" />
          </div>
        </div>
      </div>
    </Layout>
  );
}