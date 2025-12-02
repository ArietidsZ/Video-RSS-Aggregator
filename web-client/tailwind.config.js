/** @type {import('tailwindcss').Config} */
module.exports = {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                telegram: {
                    bg: '#1c252f',
                    sidebar: '#242f3d',
                    card: '#182533',
                    primary: '#5ea6e7',
                    secondary: '#8e9cb2',
                    hover: 'rgba(255, 255, 255, 0.05)',
                }
            }
        },
    },
    plugins: [],
}
