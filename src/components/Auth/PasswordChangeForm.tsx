import React, { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { useAuth } from './AuthProvider';
import toast from 'react-hot-toast';
import { Eye, EyeOff, Lock, Shield, Check } from 'lucide-react';
import styles from './authForm.module.css';

// Validation schema for password change
const passwordChangeSchema = z
  .object({
    currentPassword: z.string().min(1, 'Current password is required'),
    newPassword: z
      .string()
      .min(8, 'Password must be at least 8 characters long')
      .max(128, 'Password must be less than 128 characters')
      .regex(
        /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]/,
        'Password must contain at least one uppercase letter, one lowercase letter, one number, and one special character'
      ),
    confirmPassword: z.string().min(1, 'Please confirm your new password'),
  })
  .refine((data) => data.newPassword === data.confirmPassword, {
    message: 'Passwords do not match',
    path: ['confirmPassword'],
  });

type PasswordChangeFormData = z.infer<typeof passwordChangeSchema>;

interface PasswordChangeFormProps {
  onSuccess?: () => void;
  onCancel?: () => void;
}

export const PasswordChangeForm: React.FC<PasswordChangeFormProps> = ({ onSuccess, onCancel }) => {
  const { changePassword, loading } = useAuth();
  const [showCurrentPassword, setShowCurrentPassword] = useState(false);
  const [showNewPassword, setShowNewPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const {
    register,
    handleSubmit,
    formState: { errors },
    reset,
    watch,
  } = useForm<PasswordChangeFormData>({
    resolver: zodResolver(passwordChangeSchema),
    defaultValues: {
      currentPassword: '',
      newPassword: '',
      confirmPassword: '',
    },
    mode: 'onChange',
  });

  const newPassword = watch('newPassword');

  const onSubmit = async (data: PasswordChangeFormData) => {
    setIsSubmitting(true);
    try {
      await changePassword(data.currentPassword, data.newPassword);
      reset();
      onSuccess?.();
    } catch (error) {
      // Error is already handled in AuthProvider with toast
      console.error('Password change form submission error:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  const getPasswordStrength = (password: string): 'weak' | 'medium' | 'strong' => {
    if (password.length < 8) return 'weak';
    if (password.length < 12) return 'medium';
    if (password.length < 16) return 'strong';
    return 'strong';
  };

  const passwordStrength = getPasswordStrength(newPassword);
  const passwordStrengthColors = {
    weak: 'bg-red-500',
    medium: 'bg-yellow-500',
    strong: 'bg-green-500',
  };

  const getPasswordRequirements = (password: string) => {
    const requirements = [
      { regex: /.{8,}/, text: 'At least 8 characters', met: password.length >= 8 },
      { regex: /[A-Z]/, text: 'One uppercase letter', met: /[A-Z]/.test(password) },
      { regex: /[a-z]/, text: 'One lowercase letter', met: /[a-z]/.test(password) },
      { regex: /\d/, text: 'One number', met: /\d/.test(password) },
      { regex: /[@$!%*?&]/, text: 'One special character', met: /[@$!%*?&]/.test(password) },
    ];
    return requirements;
  };

  const requirements = getPasswordRequirements(newPassword);

  return (
    <div className={styles.authForm}>
      <div className={styles.authFormHeader}>
        <h2 className={styles.authFormTitle}>Change Password</h2>
        <p className={styles.authFormSubtitle}>
          Update your password to keep your account secure
        </p>
      </div>

      <form onSubmit={handleSubmit(onSubmit)} className={styles.authFormContent}>
        <div className={styles.formGroup}>
          <label htmlFor="currentPassword" className={styles.formLabel}>
            Current Password
          </label>
          <div className={styles.formInputWrapper}>
            <Lock className={styles.formInputIcon} size={20} />
            <input
              {...register('currentPassword')}
              type={showCurrentPassword ? 'text' : 'password'}
              id="currentPassword"
              className={`${styles.formInput} ${errors.currentPassword ? styles.error : ''}`}
              placeholder="Enter your current password"
              autoComplete="current-password"
            />
            <button
              type="button"
              onClick={() => setShowCurrentPassword(!showCurrentPassword)}
              className={styles.formInputButton}
              aria-label={showCurrentPassword ? 'Hide current password' : 'Show current password'}
            >
              {showCurrentPassword ? <EyeOff size={20} /> : <Eye size={20} />}
            </button>
          </div>
          {errors.currentPassword && (
            <p className={styles.formError}>{errors.currentPassword.message}</p>
          )}
        </div>

        <div className={styles.formGroup}>
          <label htmlFor="newPassword" className={styles.formLabel}>
            New Password
          </label>
          <div className={styles.formInputWrapper}>
            <Shield className={styles.formInputIcon} size={20} />
            <input
              {...register('newPassword')}
              type={showNewPassword ? 'text' : 'password'}
              id="newPassword"
              className={`${styles.formInput} ${errors.newPassword ? styles.error : ''}`}
              placeholder="Enter your new password"
              autoComplete="new-password"
            />
            <button
              type="button"
              onClick={() => setShowNewPassword(!showNewPassword)}
              className={styles.formInputButton}
              aria-label={showNewPassword ? 'Hide new password' : 'Show new password'}
            >
              {showNewPassword ? <EyeOff size={20} /> : <Eye size={20} />}
            </button>
          </div>
          {errors.newPassword && (
            <p className={styles.formError}>{errors.newPassword.message}</p>
          )}
          {newPassword && (
            <>
              <div className={styles.passwordStrength}>
                <div className={styles.passwordStrengthBar}>
                  <div
                    className={`${styles.passwordStrengthFill} ${passwordStrengthColors[passwordStrength]}`}
                    style={{
                      width: `${passwordStrength === 'weak' ? 33 : passwordStrength === 'medium' ? 66 : 100}%`,
                    }}
                  />
                </div>
                <span className={`${styles.passwordStrengthText} ${styles.textXs}`}>
                  Password strength: {passwordStrength}
                </span>
              </div>
              <div className={styles.passwordRequirements}>
                <p className={`${styles.textXs} ${styles.textGray600}`}>Password must contain:</p>
                <ul className={styles.requirementList}>
                  {requirements.map((req, index) => (
                    <li key={index} className={`${styles.requirementItem} ${req.met ? styles.met : styles.unmet}`}>
                      <Check size={14} className={styles.requirementIcon} />
                      <span className={styles.textXs}>{req.text}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </>
          )}
        </div>

        <div className={styles.formGroup}>
          <label htmlFor="confirmPassword" className={styles.formLabel}>
            Confirm New Password
          </label>
          <div className={styles.formInputWrapper}>
            <Lock className={styles.formInputIcon} size={20} />
            <input
              {...register('confirmPassword')}
              type={showConfirmPassword ? 'text' : 'password'}
              id="confirmPassword"
              className={`${styles.formInput} ${errors.confirmPassword ? styles.error : ''}`}
              placeholder="Confirm your new password"
              autoComplete="new-password"
            />
            <button
              type="button"
              onClick={() => setShowConfirmPassword(!showConfirmPassword)}
              className={styles.formInputButton}
              aria-label={showConfirmPassword ? 'Hide confirm password' : 'Show confirm password'}
            >
              {showConfirmPassword ? <EyeOff size={20} /> : <Eye size={20} />}
            </button>
          </div>
          {errors.confirmPassword && (
            <p className={styles.formError}>{errors.confirmPassword.message}</p>
          )}
        </div>

        <div className={styles.formActions}>
          {onCancel && (
            <button
              type="button"
              onClick={onCancel}
              className={`${styles.authFormButton} ${styles.secondaryButton}`}
              disabled={loading || isSubmitting}
            >
              Cancel
            </button>
          )}

          <button
            type="submit"
            disabled={loading || isSubmitting}
            className={styles.authFormButton}
          >
            {loading || isSubmitting ? (
              <div className={`${styles.flex} ${styles.itemsCenter} ${styles.justifyCenter}`}>
                <div className={styles.spinner} />
                Updating Password...
              </div>
            ) : (
              <div className={`${styles.flex} ${styles.itemsCenter} ${styles.justifyCenter}`}>
                <Shield size={20} className={styles.mr2} />
                Change Password
              </div>
            )}
          </button>
        </div>
      </form>
    </div>
  );
};

export default PasswordChangeForm;