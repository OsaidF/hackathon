/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

module.exports = function authPlugin(context, options) {
  return {
    name: 'auth-plugin',
    getClientModules() {
      return [
        require.resolve('./components/Auth/AuthProvider'),
        require.resolve('./components/Auth/ProtectedRoute'),
      ];
    },
    configureWebpack() {
      return {
        resolve: {
          alias: {
            '@auth': require.resolve('./lib/auth'),
            '@api': require.resolve('./lib/api'),
            '@validation': require.resolve('./lib/validation'),
          },
        },
      };
    },
  };
};