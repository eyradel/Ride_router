import { NextRequest, NextResponse } from "next/server";

const publicPaths = ["/", "/login", "/register", "/verify", "/about", "/contact"];

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;
  const token = request.cookies.get("token")?.value || (typeof window !== "undefined" ? localStorage.getItem("token") : null);

  // Allow public access to public paths
  if (publicPaths.includes(pathname)) {
    return NextResponse.next();
  }

  // Redirect unauthenticated users to /login for protected routes
  if (!token) {
    return NextResponse.redirect(new URL("/login", request.url));
  }

  return NextResponse.next();
}

export const config = {
  matcher: ["/((?!_next|api|static|favicon.ico).*)"],
}; 