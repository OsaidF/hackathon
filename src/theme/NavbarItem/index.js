import React, { useState, useEffect } from 'react';
import Link from '@docusaurus/Link';
import DefaultNavbarItem from '@theme-original/NavbarItem';
import { useAuth } from '../../components/Auth/AuthProvider';
import clsx from 'clsx';
import styles from '../../components/Auth/authForm.module.css';

function AuthButtons() {
  const { user, loading, logout } = useAuth();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  // CRITICAL: Always return the same structure during SSR and initial render
  // This prevents hydration mismatches
  if (!mounted) {
    // Return placeholder with same structure as the actual content
    return (
      <div style={{ 
        display: 'flex', 
        gap: '0.5rem', 
        alignItems: 'center',
        minWidth: '150px',
        visibility: 'hidden' // Hidden but maintains layout
      }}>
        <span>Loading</span>
      </div>
    );
  }

  // Show loading state
  if (loading) {
    return (
      <div style={{ 
        display: 'flex', 
        gap: '0.5rem', 
        alignItems: 'center',
        minWidth: '150px'
      }}>
        <span style={{ fontSize: '0.875rem', color: 'var(--ifm-color-gray-600)' }}>
          Loading...
        </span>
      </div>
    );
  }

  // Show auth buttons based on user state
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

export default function NavbarItem(props) {
  const { type } = props;
  
  if (type === 'custom-authButtons') {
    return <AuthButtons />;
  }
  
  return <DefaultNavbarItem {...props} />;
}