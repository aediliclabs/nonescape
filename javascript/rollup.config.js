import commonjs from "@rollup/plugin-commonjs";
import resolve from "@rollup/plugin-node-resolve";
import terser from "@rollup/plugin-terser";
import typescript from "@rollup/plugin-typescript";

const production = !process.env.ROLLUP_WATCH;

export default [
  // ES Module build
  {
    input: "src/index.ts",
    output: {
      file: "dist/nonescape.esm.js",
      format: "es",
      sourcemap: !production,
    },
    plugins: [
      resolve({ browser: true }),
      commonjs(),
      typescript({
        declaration: true,
        declarationDir: "dist",
        rootDir: "src",
      }),
      production && terser(),
    ].filter(Boolean),
    external: ["onnxruntime-web"],
  },

  // UMD build for script tags (unminified)
  {
    input: "src/global.ts",
    output: {
      file: "dist/nonescape.js",
      format: "umd",
      name: "Nonescape",
      sourcemap: !production,
    },
    plugins: [resolve({ browser: true }), commonjs(), typescript()].filter(Boolean),
  },

  // UMD build for script tags (minified)
  {
    input: "src/global.ts",
    output: {
      file: "dist/nonescape.min.js",
      format: "umd",
      name: "Nonescape",
      sourcemap: false,
    },
    plugins: [resolve({ browser: true }), commonjs(), typescript(), terser()].filter(Boolean),
  },
];
