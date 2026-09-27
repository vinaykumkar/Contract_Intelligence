/* Record the ContractIQ UI walkthrough (no state changes — only opens existing analyses). */
const { chromium } = require("playwright");
const OUT = "D:/ContractIQ/reports/deck_assets/video_build";
const HIGH = "0da992cdd2bd4413b0b1fa4b7637e6f8"; // demo_services_agreement (67 HIGH)
const LOW = "9113bc4ac670486984177100a502635f";  // commercial lease (LOW)
const BASE = "http://127.0.0.1:8010";

(async () => {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: { width: 1600, height: 900 },
    recordVideo: { dir: OUT, size: { width: 1600, height: 900 } },
  });
  const page = await context.newPage();

  await page.goto(BASE + "/", { waitUntil: "domcontentloaded" });
  await page.waitForTimeout(9000); // dashboard: stats + constellation

  await page.goto(`${BASE}/contracts/${HIGH}`, { waitUntil: "domcontentloaded" });
  await page.waitForTimeout(9000); // risk orb + evidence

  try {
    const card = page.getByText("Non-Compete", { exact: false }).first();
    await card.click({ timeout: 4000 });
  } catch (e) { /* keep recording even if the click target moved */ }
  await page.waitForTimeout(7000); // evidence highlight on click

  await page.mouse.wheel(0, 700);
  await page.waitForTimeout(8000); // findings panel

  await page.goto(BASE + "/contracts", { waitUntil: "domcontentloaded" });
  await page.waitForTimeout(6000); // history

  await page.goto(`${BASE}/contracts/${LOW}`, { waitUntil: "domcontentloaded" });
  await page.waitForTimeout(8000); // low-risk orb

  await page.goto(BASE + "/intelligence", { waitUntil: "domcontentloaded" });
  await page.waitForTimeout(8000); // model state + methodology

  await context.close();
  await browser.close();
  console.log("recording saved");
})().catch((e) => { console.error(e); process.exit(1); });
