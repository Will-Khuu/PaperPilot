import styles from "./icon.module.css";

export function Icon() {
  return <span className={styles.icon}><svg viewBox="0 0 24 24" aria-hidden="true">
    <path d="m12 2 9 5v10l-9 5-9-5V7Z" fill="currentColor" />
    <path d="m3 7 9 5 9-5M12 12v10M7.5 4.5l9 5v5" fill="none" stroke="white" strokeWidth="1.5" strokeLinejoin="round" />
  </svg></span>;
}
export function PdfBadge() { return <span className={styles.pdf} aria-hidden="true">PDF</span>; }
