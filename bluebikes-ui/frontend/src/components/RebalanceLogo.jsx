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
        <radialGradient id="glow" cx="50%" cy="50%" r="70%">
          <stop offset="0%" stopColor="#00e5ff" stopOpacity="0.8" />
          <stop offset="100%" stopColor="#003b46" stopOpacity="0" />
        </radialGradient>

        <linearGradient id="neon" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#18ffff" />
          <stop offset="100%" stopColor="#00bcd4" />
        </linearGradient>

        <linearGradient id="balanceBeam" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#00bcd4" />
          <stop offset="100%" stopColor="#18ffff" />
        </linearGradient>
      </defs>

      <circle cx="120" cy="120" r="110" fill="url(#glow)" />

      <circle
        cx="120"
        cy="120"
        r="95"
        stroke="url(#neon)"
        strokeWidth="4"
        fill="none"
      />

      <line
        x1="40"
        y1="120"
        x2="200"
        y2="120"
        stroke="url(#balanceBeam)"
        strokeWidth="6"
        strokeLinecap="round"
        opacity="0.8"
      />
      <line
        x1="120"
        y1="40"
        x2="120"
        y2="200"
        stroke="url(#balanceBeam)"
        strokeWidth="6"
        strokeLinecap="round"
        opacity="0.6"
      />

      <g
        fill="none"
        stroke="#18ffff"
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

      <text
        x="120"
        y="222"
        fontFamily="Inter, sans-serif"
        fontSize="16"
        fill="#18ffff"
        textAnchor="middle"
        opacity="0.9"
        letterSpacing="2"
      >
        REBALANCE AI {/* TODO:change name*/}
      </text> 
    </svg>
  );
};

export default RebalanceLogo;