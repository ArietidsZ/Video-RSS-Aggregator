import React from 'react';
import { ExclamationTriangleIcon } from '@heroicons/react/24/outline';

function ErrorMessage({ title = 'Error', message, onRetry, className = '' }) {
  return (
    <div className={`text-center py-12 ${className}`}>
      <ExclamationTriangleIcon className="mx-auto h-12 w-12 text-red-400" />
      <h3 className="mt-2 text-sm font-medium text-gray-900">{title}</h3>
      <p className="mt-1 text-sm text-gray-500">{message}</p>
      {onRetry && (
        <div className="mt-6">
          <button
            type="button"
            onClick={onRetry}
            className="btn btn-primary"
          >
            Try again
          </button>
        </div>
      )}
    </div>
  );
}

export default ErrorMessage;