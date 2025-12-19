import React, { useState, useEffect } from 'react';
import DefaultNavbarItem from '@theme-original/NavbarItem';
import { useAuth } from '../../components/Auth/AuthProvider';
import styles from './styles.module.css';

function AuthButtons() {
  const [mounted, setMounted] = useState(false);
  const { user, loading, logout } = useAuth();

  useEffect(() => {
    setMounted(true);
  }, []);

  // Return consistent structure during SSR and initial client render
  if (!mounted || loading) {
    return <div className={styles.authContainer} />;
  }

  return (
    <div className={styles.authContainer}>
      {user ? (
        <>
          <span className={styles.userName}>
            {user.name || user.email}
          </span>
          <button
            className={styles.authButton}
            onClick={() => logout()}
          >
            Sign Out
          </button>
        </>
      ) : (
        <>
          <a href="/auth/login" className={styles.authLink}>
            Sign In
          </a>
          <a href="/auth/register" className={styles.authButton}>
            Sign Up
          </a>
        </>
      )}
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