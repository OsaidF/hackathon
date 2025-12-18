import React, { useState, useEffect } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { useAuth } from './AuthProvider';
import toast from 'react-hot-toast';
import { User, Camera, Save, X } from 'lucide-react';
import styles from './authForm.module.css';

// Validation schema for profile settings
const profileSettingsSchema = z.object({
  name: z.string().min(2, 'Name must be at least 2 characters').max(255, 'Name must be less than 255 characters'),
  image: z.string().url('Please enter a valid image URL').optional().or(z.literal('')),
});

type ProfileSettingsFormData = z.infer<typeof profileSettingsSchema>;

interface ProfileSettingsProps {
  onSuccess?: () => void;
  onCancel?: () => void;
}

export const ProfileSettings: React.FC<ProfileSettingsProps> = ({ onSuccess, onCancel }) => {
  const { user, updateProfile } = useAuth();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [currentProfile, setCurrentProfile] = useState(null);

  const {
    register,
    handleSubmit,
    formState: { errors, isDirty },
    reset,
    watch,
    setValue,
  } = useForm<ProfileSettingsFormData>({
    resolver: zodResolver(profileSettingsSchema),
    defaultValues: {
      name: '',
      image: '',
    },
  });

  // Load current profile data
  useEffect(() => {
    if (user) {
      const profileData = {
        name: user.name || '',
        image: user.image || '',
      };
      setCurrentProfile(profileData);
      reset(profileData);
    }
  }, [user, reset]);

  const onSubmit = async (data: ProfileSettingsFormData) => {
    if (!isDirty) {
      toast.info('No changes to save');
      return;
    }

    setIsSubmitting(true);
    try {
      // Convert empty string to null for image URL
      const submitData = {
        ...data,
        image: data.image || null,
      };

      await updateProfile(submitData);

      setCurrentProfile(submitData);
      reset(submitData);

      toast.success('Profile updated successfully!');
      onSuccess?.();
    } catch (error) {
      // Error is already handled in AuthProvider with toast
      console.error('Profile settings submission error:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleCancel = () => {
    if (currentProfile) {
      reset(currentProfile);
    }
    onCancel?.();
  };

  const watchedImage = watch('image');

  return (
    <div className={styles.authForm}>
      <div className={styles.authFormHeader}>
        <h2 className={styles.authFormTitle}>Profile Settings</h2>
        <p className={styles.authFormSubtitle}>
          Update your personal information
        </p>
      </div>

      <form onSubmit={handleSubmit(onSubmit)} className={styles.authFormContent}>
        <div className={styles.formGroup}>
          <label htmlFor="name" className={styles.formLabel}>
            Full Name
          </label>
          <div className={styles.formInputWrapper}>
            <User className={styles.formInputIcon} size={20} />
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
          <label htmlFor="image" className={styles.formLabel}>
            Profile Picture URL
          </label>
          <div className={styles.formInputWrapper}>
            <Camera className={styles.formInputIcon} size={20} />
            <input
              {...register('image')}
              type="url"
              id="image"
              className={`${styles.formInput} ${errors.image ? styles.error : ''}`}
              placeholder="https://example.com/your-image.jpg"
              autoComplete="url"
            />
          </div>
          {errors.image && (
            <p className={styles.formError}>{errors.image.message}</p>
          )}
          {watchedImage && (
            <div className={styles.imagePreview}>
              <img
                src={watchedImage}
                alt="Profile preview"
                className={styles.imagePreviewImg}
                onError={(e) => {
                  e.currentTarget.style.display = 'none';
                }}
              />
            </div>
          )}
        </div>

        <div className={styles.profileInfo}>
          <div className={styles.profileInfoItem}>
            <span className={styles.profileInfoLabel}>Email:</span>
            <span className={styles.profileInfoValue}>{user?.email}</span>
          </div>
          <div className={styles.profileInfoItem}>
            <span className={styles.profileInfoLabel}>Account Status:</span>
            <span className={`${styles.profileInfoValue} ${user?.emailVerified ? styles.verified : styles.unverified}`}>
              {user?.emailVerified ? 'Verified' : 'Not Verified'}
            </span>
          </div>
          <div className={styles.profileInfoItem}>
            <span className={styles.profileInfoLabel}>Member Since:</span>
            <span className={styles.profileInfoValue}>
              {user?.createdAt ? new Date(user.createdAt).toLocaleDateString() : 'N/A'}
            </span>
          </div>
        </div>

        <div className={styles.formActions}>
          {onCancel && (
            <button
              type="button"
              onClick={handleCancel}
              className={`${styles.authFormButton} ${styles.secondaryButton}`}
              disabled={isSubmitting}
            >
              <X size={20} className={styles.mr2} />
              Cancel
            </button>
          )}

          <button
            type="submit"
            disabled={isSubmitting || !isDirty}
            className={styles.authFormButton}
          >
            {isSubmitting ? (
              <div className={`${styles.flex} ${styles.itemsCenter} ${styles.justifyCenter}`}>
                <div className={styles.spinner} />
                Saving...
              </div>
            ) : (
              <div className={`${styles.flex} ${styles.itemsCenter} ${styles.justifyCenter}`}>
                <Save size={20} className={styles.mr2} />
                Save Changes
              </div>
            )}
          </button>
        </div>
      </form>
    </div>
  );
};

export default ProfileSettings;