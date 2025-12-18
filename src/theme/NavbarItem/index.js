import React from 'react';
import ExecutionEnvironment from '@docusaurus/ExecutionEnvironment';
import DefaultNavbarItem from '@theme-original/NavbarItem';
import { useAuth } from '../../components/Auth/AuthProvider';
import styles from './styles.module.css';

function AuthButtons() {
  // Don't render during SSR
  if (!ExecutionEnvironment.canUseDOM) {
    return null;
  }

  const { user, loading, logout } = useAuth();

  if (loading) return null;

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