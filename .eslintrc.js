module.exports = {
  root: true,
  env: {
    browser: true,
    es2020: true,
    node: true,
  },
  extends: [
    'eslint:recommended',
    'plugin:react/recommended',
    'plugin:react-hooks/recommended',
  ],
  ignorePatterns: ['dist', '.eslintrc.js', 'build'],
  parserOptions: {
    ecmaVersion: 'latest',
    sourceType: 'module',
    ecmaFeatures: {
      jsx: true,
    },
  },
  settings: {
    react: {
      version: 'detect',
    },
  },
  plugins: ['react-refresh'],
  rules: {
    'react-refresh/only-export-components': [
      'warn',
      { allowConstantExport: true },
    ],
    'react/prop-types': 'off', // Docusaurus components often don't need explicit prop-types
    'react/react-in-jsx-scope': 'off', // Not needed with React 17+
  },
  overrides: [
    {
      // Special rules for MDX files
      files: ['*.mdx'],
      extends: ['plugin:mdx/recommended'],
      rules: {
        'mdx/no-unescaped-entities': 'off',
        'mdx/no-unused-expressions': 'off',
      },
    },
  ],
};