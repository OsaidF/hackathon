import React, { useState, useEffect } from 'react';
import Link from '@docusaurus/Link';
import DefaultNavbarItem from '@theme-original/NavbarItem';
import { useAuth } from '../../components/Auth/AuthProvider';

function AuthButtons() {
  const { user, loading, logout } = useAuth();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  // Always return a container to maintain consistent DOM structure
  if (!mounted || loading) {
    return (
      <div 
        style={{ 
          minWidth: '150px', 
          height: '36px',
          display: 'inline-block'
        }} 
      />
    );
  }

  if (!user) {
    return (
      <div style={{ display: 'flex', gap: '0.5rem' }}>
        <Link to="/auth/login" className="navbar__link">
          Sign In
        </Link>
        <Link 
          to="/auth/register" 
          className="navbar__link"
          style={{
            backgroundColor: 'var(--ifm-color-primary)',
            color: 'white',
            padding: '0.5rem 1rem',
            borderRadius: '0.375rem'
          }}
        >
          Sign Up
        </Link>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', gap: '0.5rem' }}>
      <Link to="/auth/profile" className="navbar__link">
        Profile
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