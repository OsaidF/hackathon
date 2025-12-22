import React, { useState, useEffect } from 'react';
import { useHistory } from '@docusaurus/router'
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { useAuth } from './AuthProvider';
import toast from 'react-hot-toast';
import { Eye, EyeOff, Mail, Lock, UserPlus } from 'lucide-react';
import styles from './authForm.module.css';

// Validation schema
const registerSchema = z
  .object({
    name: z
      .string()
      .min(2, 'Name must be at least 2 characters')
      .max(50, 'Name must be less than 50 characters'),
    email: z.string().email('Please enter a valid email address'),
    password: z
      .string()
      .min(8, 'Password must be at least 8 characters long')
      .max(128, 'Password must be less than 128 characters')
      .regex(
        /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]+$/,
        'Password must contain at least one uppercase letter, one lowercase letter, one number, and one special character'
      ),
    confirmPassword: z.string().min(1, 'Please confirm your password'),
  })
  .refine((data) => data.password === data.confirmPassword, {
    message: 'Passwords do not match',
    path: ['confirmPassword'],
  });

type RegisterFormData = z.infer<typeof registerSchema>;

interface RegisterFormProps {
  onSuccess?: () => void;
  redirectTo?: string;
}

export const RegisterForm: React.FC<RegisterFormProps> = ({ onSuccess, redirectTo }) => {
  const { register: registerUser, loading, user } = useAuth();
  const history = useHistory()
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [mounted, setMounted] = useState(false);
  
  useEffect(() => {
    setMounted(true);
  }, []);


  useEffect(() => {
      console.log('AuthButtons - mounted:', mounted, 'loading:', loading, 'user:', user);
    }, [mounted, loading, user]);
  const {
    register,
    handleSubmit,
    formState: { errors },
    reset,
    watch,
  } = useForm<RegisterFormData>({
    resolver: zodResolver(registerSchema),
    defaultValues: {
      name: '',
      email: '',
      password: '',
      confirmPassword: '',
    },
    mode: 'onChange',
  });

  const password = watch('password');

  const onSubmit = async (data: RegisterFormData) => {
    setIsSubmitting(true);
    try {
      await registerUser(data.email, data.password, data.name);
      reset();
      onSuccess?.();

      // Redirect if specified
      if (redirectTo) {
        history.push(redirectTo);
      }
    } catch (error) {
      // Error is already handled in AuthProvider with toast
      console.error('Registration form submission error:', error);
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

  const passwordStrength = getPasswordStrength(password);
  const passwordStrengthColors = {
    weak: 'bg-red-500',
    medium: 'bg-yellow-500',
    strong: 'bg-green-500',
  };

  return (
    <div className={styles.authForm}>
      <div className={styles.authFormHeader}>
        <h2 className={styles.authFormTitle}>Create Account</h2>
        <p className={styles.authFormSubtitle}>
          Join the Humanoid Robotics Lab Guide community
        </p>
      </div>

      <form onSubmit={handleSubmit(onSubmit)} className={styles.authFormContent}>
        <div className={styles.formGroup}>
          <label htmlFor="name" className={styles.formLabel}>
            Full Name
          </label>
          <div className={styles.formInputWrapper}>
            <UserPlus className={styles.formInputIcon} size={20} />
            <input
              {...register('name')}
              type="text"
              id="name"
              className={`${styles.formInput} ${errors.name ? styles.error : ''}`}
              placeholder="Enter your full name"
              autoComplete="name"
            />
          </div>
          {errors.name && (
            <p className={styles.formError}>{errors.name.message}</p>
          )}
        </div>

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
              autoComplete="new-password"
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
          {password && (
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
          )}
        </div>

        <div className={styles.formGroup}>
          <label htmlFor="confirmPassword" className={styles.formLabel}>
            Confirm Password
          </label>
          <div className={styles.formInputWrapper}>
            <Lock className={styles.formInputIcon} size={20} />
            <input
              {...register('confirmPassword')}
              type={showConfirmPassword ? 'text' : 'password'}
              id="confirmPassword"
              className={`${styles.formInput} ${errors.confirmPassword ? styles.error : ''}`}
              placeholder="Confirm your password"
              autoComplete="new-password"
            />
            <button
              type="button"
              onClick={() => setShowConfirmPassword(!showConfirmPassword)}
              className="form-input-button"
              aria-label={showConfirmPassword ? 'Hide password' : 'Show password'}
            >
              {showConfirmPassword ? <EyeOff size={20} /> : <Eye size={20} />}
            </button>
          </div>
          {errors.confirmPassword && (
            <p className={styles.formError}>{errors.confirmPassword.message}</p>
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
              Creating Account...
            </div>
          ) : (
            <div className={`${styles.flex} ${styles.itemsCenter} ${styles.justifyCenter}`}>
              <UserPlus size={20} className={styles.mr2} />
              Create Account
            </div>
          )}
        </button>

        <div className={styles.authFormFooter}>
          <p className={`${styles.textSm} ${styles.textGray600}`}>
            Already have an account?{' '}
            <a href="/auth/login" className={styles.authLink}>
              Sign in
            </a>
          </p>
        </div>
      </form>
    </div>
  );
};

export default RegisterForm;