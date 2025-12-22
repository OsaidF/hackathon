import React, { useState, useEffect } from 'react';
import DefaultNavbarItem from '@theme-original/NavbarItem';
import Link from '@docusaurus/Link';
import { useAuth } from '../../components/Auth/AuthProvider';
import ExecutionEnvironment from '@docusaurus/ExecutionEnvironment';

function AuthButtons() {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  // Don't render during SSR
  if (!ExecutionEnvironment.canUseDOM || !mounted) {
    return <div style={{ minWidth: '150px', height: '36px' }} />;
  }

  // Now safe to use hooks
  return <AuthButtonsClient />;
}

function AuthButtonsClient() {
  const { user, loading, logout } = useAuth();

  if (loading) {
    return <div style={{ minWidth: '150px', height: '36px' }} />;
  }

  if (!user) {
    return (
      <div style={{ display: 'flex', gap: '0.5rem' }}>
        <Link to="/auth/login" className="navbar__link">
          Sign In
        </Link>
        <Link 
          to="/auth/register" 
          className="navbar__link button button--primary"
        >
          Sign Up
        </Link>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', gap: '0.5rem' }}>
      <Link to="/auth/profile" className="navbar__link">
        {user.name}
      </Link>
      <button
        className="navbar__link"
        onClick={() => logout()}
        style={{
          background: 'none',
          border: 'none',
          cursor: 'pointer'
        }}
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