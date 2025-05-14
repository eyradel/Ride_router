import NextAuth from "next-auth";
import CredentialsProvider from "next-auth/providers/credentials";
import axios from "axios";

export default NextAuth({
  providers: [
    CredentialsProvider({
      name: "Credentials",
      credentials: {
        username: { label: "Username", type: "text" },
        password: { label: "Password", type: "password" }
      },
      async authorize(credentials) {
        try {
          const res = await axios.post("http://localhost:8000/auth/token", {
            username: credentials.username,
            password: credentials.password,
          });
          const user = res.data;
          if (user && user.access_token) {
            return { ...user, name: credentials.username };
          }
          return null;
        } catch (e) {
          return null;
        }
      }
    })
  ],
  session: {
    strategy: "jwt",
  },
  callbacks: {
    async jwt({ token, user }) {
      if (user) {
        token.accessToken = user.access_token;
      }
      return token;
    },
    async session({ session, token }) {
      session.accessToken = token.accessToken;
      return session;
    }
  },
  pages: {
    signIn: "/login",
  }
}); 