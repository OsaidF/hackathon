import React, { useState } from 'react';
import Layout from '@theme/Layout';
import { useAuth } from '../../components/Auth/AuthProvider';
import ProfileSettings from '../../components/Auth/ProfileSettings';
import PasswordChangeForm from '../../components/Auth/PasswordChangeForm';
import EmailVerification from '../../components/Auth/EmailVerification';
import clsx from 'clsx';
import styles from './auth.module.css';
import { User, Shield, Mail, LogOut, ChevronRight } from 'lucide-react';

export default function ProfilePage() {
  const { user, logout } = useAuth();
  const [activeTab, setActiveTab] = useState('profile');

  if (!user) {
    return (
      <Layout
        title="Profile"
        description="Your profile information"
      >
        <div className={styles.authPage}>
          <div className={styles.authContainer}>
            <div className={styles.authCard}>
              <div className={styles.authHeader}>
                <h1 className={styles.authTitle}>Authentication Required</h1>
                <p className={styles.authSubtitle}>
                  Please sign in to view your profile
                </p>
              </div>
              <div className={styles.authForm}>
                <a href="/auth/login" className={styles.authFormButton}>
                  Sign In
                </a>
              </div>
            </div>
          </div>
        </div>
      </Layout>
    );
  }

  const tabs = [
    {
      id: 'profile',
      label: 'Profile Settings',
      icon: User,
      component: ProfileSettings,
    },
    {
      id: 'password',
      label: 'Password & Security',
      icon: Shield,
      component: PasswordChangeForm,
    },
    {
      id: 'verification',
      label: 'Email Verification',
      icon: Mail,
      component: EmailVerification,
    },
  ];

  const ActiveComponent = tabs.find(tab => tab.id === activeTab)?.component;

  return (
    <Layout
      title="Profile"
      description="Manage your account settings and preferences"
    >
      <div className={styles.profilePage}>
        <div className={styles.profileContainer}>
          {/* User Profile Header */}
          <div className={styles.profileHeader}>
            <div className={styles.profileAvatar}>
              {user.image ? (
                <img
                  src={user.image}
                  alt={user.name || 'Profile'}
                  className={styles.avatarImage}
                />
              ) : (
                <div className={styles.avatarPlaceholder}>
                  <User size={32} />
                </div>
              )}
            </div>
            <div className={styles.profileInfo}>
              <h1 className={styles.profileName}>
                {user.name || 'User'}
              </h1>
              <p className={styles.profileEmail}>{user.email}</p>
              <div className={styles.profileStatus}>
                <span className={clsx(
                  styles.statusBadge,
                  user.emailVerified ? styles.verified : styles.unverified
                )}>
                  {user.emailVerified ? 'Verified' : 'Not Verified'}
                </span>
                <span className={styles.memberSince}>
                  Member since {new Date(user.createdAt).toLocaleDateString()}
                </span>
              </div>
            </div>
            <div className={styles.profileActions}>
              <button
                onClick={() => logout()}
                className={clsx(styles.profileActionButton, styles.logoutButton)}
              >
                <LogOut size={16} />
                Sign Out
              </button>
            </div>
          </div>

          {/* Navigation Tabs */}
          <div className={styles.profileTabs}>
            <nav className={styles.tabNav}>
              {tabs.map((tab) => {
                const Icon = tab.icon;
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={clsx(
                      styles.tabButton,
                      activeTab === tab.id && styles.tabButtonActive
                    )}
                  >
                    <Icon size={18} />
                    {tab.label}
                    <ChevronRight
                      size={16}
                      className={clsx(
                        styles.tabChevron,
                        activeTab === tab.id && styles.tabChevronActive
                      )}
                    />
                  </button>
                );
              })}
            </nav>
          </div>

          {/* Tab Content */}
          <div className={styles.profileContent}>
            <div className={styles.contentCard}>
              {ActiveComponent && <ActiveComponent />}
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
}