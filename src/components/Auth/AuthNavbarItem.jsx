import React from 'react';
import Link from '@docusaurus/Link';
import { useAuth } from './AuthProvider';
import clsx from 'clsx';
import styles from './authForm.module.css';

export default function AuthNavbarItem() {
  const { user, loading, logout } = useAuth();

  if (loading) {
    return null; // Don't show anything while loading
  }

  if (!user) {
    return (
      <>
        <Link
          to="/auth/login"
          className={clsx('navbar__link', styles.authNavButton)}
        >
          Sign In
        </Link>
        <Link
          to="/auth/register"
          className={clsx('navbar__link', styles.authNavButton, styles.authNavButtonPrimary)}
        >
          Sign Up
        </Link>
      </>
    );
  }

  return (
    <div className={styles.authUserMenu}>
      <Link
        to="/auth/profile"
        className={clsx('navbar__link', styles.authNavButton)}
      >
        Profile
      </Link>
      <button
        className={clsx('navbar__link', styles.authNavButton)}
        onClick={() => logout()}
      >
        Sign Out
      </button>
    </div>
  );
}