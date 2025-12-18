import React from 'react';
import { AuthProvider } from '../components/Auth/AuthProvider';
import { Toaster } from 'react-hot-toast';

// Default implementation, that you can customize
export default function Root({children}) {
  return <>
    <AuthProvider>
        {children}
         <Toaster
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: 'var(--ifm-background-color)',
            color: 'var(--ifm-color-emphasis-900)',
            border: '1px solid var(--ifm-color-emphasis-200)',
            fontSize: '0.875rem',
          },
          success: {
            iconTheme: {
              primary: 'var(--ifm-color-success)',
              secondary: '#fff',
            },
          },
          error: {
            iconTheme: {
              primary: 'var(--ifm-color-danger)',
              secondary: '#fff',
            },
          },
        }}
      />
    </AuthProvider>
  </>;
}