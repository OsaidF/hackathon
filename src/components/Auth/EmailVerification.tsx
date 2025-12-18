import React, { useState } from 'react';
import { useAuth } from './AuthProvider';
import toast from 'react-hot-toast';
import { Mail, Send, CheckCircle, AlertCircle, RefreshCw } from 'lucide-react';
import styles from './authForm.module.css';

export const EmailVerification: React.FC = () => {
  const { user, sendEmailVerification } = useAuth();
  const [isSending, setIsSending] = useState(false);
  const [lastSent, setLastSent] = useState<Date | null>(null);

  const handleSendVerification = async () => {
    if (lastSent && Date.now() - lastSent.getTime() < 60000) {
      const remainingTime = Math.ceil((60000 - (Date.now() - lastSent.getTime())) / 1000);
      toast.error(`Please wait ${remainingTime} seconds before sending another verification email.`);
      return;
    }

    setIsSending(true);
    try {
      await sendEmailVerification();
      setLastSent(new Date());
    } catch (error) {
      // Error is already handled in AuthProvider with toast
      console.error('Email verification error:', error);
    } finally {
      setIsSending(false);
    }
  };

  const canResend = !lastSent || Date.now() - lastSent.getTime() >= 60000;
  const remainingTime = lastSent ? Math.ceil((60000 - (Date.now() - lastSent.getTime())) / 1000) : 0;

  if (!user) {
    return null;
  }

  return (
    <div className={styles.authForm}>
      <div className={styles.authFormHeader}>
        <h2 className={styles.authFormTitle}>Email Verification</h2>
        <p className={styles.authFormSubtitle}>
          Verify your email address to access all features
        </p>
      </div>

      <div className={styles.verificationContent}>
        <div className={styles.verificationStatus}>
          {user.emailVerified ? (
            <div className={styles.verificationSuccess}>
              <CheckCircle className={styles.verificationIcon} size={48} />
              <h3 className={styles.verificationTitle}>Email Verified</h3>
              <p className={styles.verificationMessage}>
                Your email address <strong>{user.email}</strong> has been verified.
              </p>
            </div>
          ) : (
            <div className={styles.verificationPending}>
              <AlertCircle className={styles.verificationIcon} size={48} />
              <h3 className={styles.verificationTitle}>Email Verification Required</h3>
              <p className={styles.verificationMessage}>
                We've sent a verification email to <strong>{user.email}</strong>.
                Please check your inbox and click the verification link to complete your registration.
              </p>

              <div className={styles.verificationActions}>
                <button
                  onClick={handleSendVerification}
                  disabled={!canResend || isSending}
                  className={`${styles.authFormButton} ${styles.verificationButton}`}
                >
                  {isSending ? (
                    <>
                      <RefreshCw className={styles.spinningIcon} size={20} />
                      Sending...
                    </>
                  ) : canResend ? (
                    <>
                      <Send size={20} className={styles.mr2} />
                      Resend Verification Email
                    </>
                  ) : (
                    <>
                      <Mail size={20} className={styles.mr2} />
                      Resend in {remainingTime}s
                    </>
                  )}
                </button>
              </div>

              {!canResend && (
                <div className={styles.verificationNote}>
                  <p className={styles.noteText}>
                    Verification emails can only be sent once per minute to prevent spam.
                  </p>
                </div>
              )}
            </div>
          )}
        </div>

        <div className={styles.verificationTips}>
          <h4 className={styles.tipsTitle}>Tips:</h4>
          <ul className={styles.tipsList}>
            <li className={styles.tipItem}>
              Check your spam or junk folder if you don't see the verification email
            </li>
            <li className={styles.tipItem}>
              Make sure the email address we have on file is correct
            </li>
            <li className={styles.tipItem}>
              Verification links expire after 24 hours
            </li>
            <li className={styles.tipItem}>
              Contact support if you're having trouble verifying your email
            </li>
          </ul>
        </div>

        <div className={styles.verificationHelp}>
          <h4 className={styles.helpTitle}>Need Help?</h4>
          <p className={styles.helpText}>
            If you're experiencing issues with email verification, please check your
            email settings or contact our support team for assistance.
          </p>
        </div>
      </div>
    </div>
  );
};

export default EmailVerification;