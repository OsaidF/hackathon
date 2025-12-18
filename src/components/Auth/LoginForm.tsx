import React, { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { useAuth } from './AuthProvider';
import toast from 'react-hot-toast';
import { Eye, EyeOff, Mail, Lock, LogIn } from 'lucide-react';
import styles from './authForm.module.css';

// Validation schema
const loginSchema = z.object({
  email: z.string().email('Please enter a valid email address'),
  password: z.string().min(1, 'Password is required'),
});

type LoginFormData = z.infer<typeof loginSchema>;

interface LoginFormProps {
  onSuccess?: () => void;
  redirectTo?: string;
}

export const LoginForm: React.FC<LoginFormProps> = ({ onSuccess, redirectTo }) => {
  const { login, loading } = useAuth();
  const [showPassword, setShowPassword] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const {
    register,
    handleSubmit,
    formState: { errors },
    reset,
  } = useForm<LoginFormData>({
    resolver: zodResolver(loginSchema),
    defaultValues: {
      email: '',
      password: '',
    },
    mode: 'onChange',
  });

  const onSubmit = async (data: LoginFormData) => {
    setIsSubmitting(true);
    try {
      await login(data.email, data.password);
      reset();
      onSuccess?.();

      // Redirect if specified
      if (redirectTo) {
        window.location.href = redirectTo;
      }
    } catch (error) {
      // Error is already handled in AuthProvider with toast
      console.error('Login form submission error:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className={styles.authForm}>
      <div className={styles.authFormHeader}>
        <h2 className={styles.authFormTitle}>Welcome Back</h2>
        <p className={styles.authFormSubtitle}>
          Sign in to your Humanoid Robotics Lab Guide account
        </p>
      </div>

      <form onSubmit={handleSubmit(onSubmit)} className={styles.authFormContent}>
        <div className={styles.formGroup}>
          <label htmlFor="email" className={styles.formLabel}>
            Email Address
          </label>
          <div className={styles.formInputWrapper}>
            <Mail className={styles.formInputIcon} size={20} />
            <input
              {...register('email')}
              type="email"
              id="email"
              className={`${styles.formInput} ${errors.email ? styles.error : ''}`}
              placeholder="Enter your email"
              autoComplete="email"
            />
          </div>
          {errors.email && (
            <p className={styles.formError}>{errors.email.message}</p>
          )}
        </div>

        <div className={styles.formGroup}>
          <label htmlFor="password" className={styles.formLabel}>
            Password
          </label>
          <div className={styles.formInputWrapper}>
            <Lock className={styles.formInputIcon} size={20} />
            <input
              {...register('password')}
              type={showPassword ? 'text' : 'password'}
              id="password"
              className={`${styles.formInput} ${errors.password ? styles.error : ''}`}
              placeholder="Enter your password"
              autoComplete="current-password"
            />
            <button
              type="button"
              onClick={() => setShowPassword(!showPassword)}
              className={styles.formInputButton}
              aria-label={showPassword ? 'Hide password' : 'Show password'}
            >
              {showPassword ? <EyeOff size={20} /> : <Eye size={20} />}
            </button>
          </div>
          {errors.password && (
            <p className={styles.formError}>{errors.password.message}</p>
          )}
        </div>

        <button
          type="submit"
          disabled={loading || isSubmitting}
          className={styles.authFormButton}
        >
          {loading || isSubmitting ? (
            <div className={`${styles.flex} ${styles.itemsCenter} ${styles.justifyCenter}`}>
              <div className={styles.spinner} />
              Signing In...
            </div>
          ) : (
            <div className={`${styles.flex} ${styles.itemsCenter} ${styles.justifyCenter}`}>
              <LogIn size={20} className={styles.mr2} />
              Sign In
            </div>
          )}
        </button>

        <div className={styles.authFormFooter}>
          <p className={`${styles.textSm} ${styles.textGray600}`}>
            Don't have an account?{' '}
            <a href="/auth/register" className={styles.authLink}>
              Sign up
            </a>
          </p>
        </div>
      </form>
    </div>
  );
};

export default LoginForm;