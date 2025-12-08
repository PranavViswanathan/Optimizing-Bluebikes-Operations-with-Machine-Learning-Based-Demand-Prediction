import React from "react";

const RebalanceLogo = ({ size = 240 }) => {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 240 240"
      xmlns="http://www.w3.org/2000/svg"
    >
      <defs>

        {/* Outer glow softly tinted with new highlight */}
        <radialGradient id="glow" cx="50%" cy="50%" r="70%">
          <stop offset="0%" stopColor="#87CBF8" stopOpacity="0.9" />
          <stop offset="100%" stopColor="#0f172a" stopOpacity="0" />
        </radialGradient>

        {/* Neon ring gradient updated to match design system */}
        <linearGradient id="neon" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#87CBF8" />
          <stop offset="100%" stopColor="#0ea5e9" />
        </linearGradient>

        {/* Crosshair / balance-beam updated for theme cohesion */}
        <linearGradient id="balanceBeam" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#87CBF8" />
          <stop offset="50%" stopColor="#7dd3fc" />
          <stop offset="100%" stopColor="#0ea5e9" />
        </linearGradient>

      </defs>

      {/* Outer glow */}
      <circle cx="120" cy="120" r="110" fill="url(#glow)" />

      {/* Neon ring */}
      <circle
        cx="120"
        cy="120"
        r="95"
        stroke="url(#neon)"
        strokeWidth="4"
        fill="none"
      />

      {/* Crosshair beam */}
      <line
        x1="40"
        y1="120"
        x2="200"
        y2="120"
        stroke="url(#balanceBeam)"
        strokeWidth="6"
        strokeLinecap="round"
        opacity="0.9"
      />
      <line
        x1="120"
        y1="40"
        x2="120"
        y2="200"
        stroke="url(#balanceBeam)"
        strokeWidth="6"
        strokeLinecap="round"
        opacity="0.7"
      />

      {/* Bike lines — recolored to unified neon highlight */}
      <g
        fill="none"
        stroke="#87CBF8"
        strokeWidth="6"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <circle cx="85" cy="150" r="22" />
        <circle cx="155" cy="150" r="22" />

        <line x1="85" y1="150" x2="115" y2="150" />
        <line x1="115" y1="150" x2="140" y2="120" />
        <line x1="85" y1="150" x2="110" y2="110" />
        <line x1="110" y1="110" x2="140" y2="120" />
        <line x1="140" y1="120" x2="155" y2="150" />

        <line x1="140" y1="120" x2="160" y2="100" />
        <line x1="160" y1="100" x2="170" y2="105" />

        <line x1="110" y1="110" x2="100" y2="100" />
        <line x1="100" y1="100" x2="120" y2="100" />
      </g>

      {/* Label text */}
      <text
        x="120"
        y="222"
        fontFamily="Inter, sans-serif"
        fontSize="20"
        fill="#87CBF8"
        textAnchor="middle"
        opacity="0.95"
        letterSpacing="2"
      >
        REBALANCE AI {/*TODO: rename*/}
      </text>
    </svg>
  );
};

export default RebalanceLogo;