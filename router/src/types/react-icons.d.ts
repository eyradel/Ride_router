declare module 'react-icons/fi' {
  import { FC, SVGAttributes } from 'react';
  
  export interface IconBaseProps extends SVGAttributes<SVGElement> {
    size?: string | number;
    color?: string;
    title?: string;
  }
  
  export type IconType = FC<IconBaseProps>;
  
  export const FiMail: IconType;
  export const FiUser: IconType;
  export const FiPhone: IconType;
  export const FiGlobe: IconType;
  export const FiBriefcase: IconType;
  export const FiLock: IconType;
} 