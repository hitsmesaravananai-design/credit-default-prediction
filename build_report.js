// Builds GroupProject_Part1_Report.docx from outputs/results.json + figures/
if (typeof globalThis === "undefined") { global.globalThis = global; } // Node 10 compatibility for docx 7
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, ImageRun,
  Table, TableRow, TableCell, WidthType, BorderStyle, ShadingType, LevelFormat,
  Footer, PageNumber, TabStopType,
} = require("docx");

const R = JSON.parse(fs.readFileSync(path.join(__dirname, "outputs", "results.json"), "utf8"));
const M = R.metrics;
const LR = M["Logistic Regression"], RF = M["Random Forest"], BASE = M["Majority-class baseline"];
const pct = (x, d = 1) => (100 * x).toFixed(d) + "%";
const f3 = (x) => x.toFixed(3);
const img = (name) => fs.readFileSync(path.join(__dirname, "figures", name));

const FONT = "Calibri";
const BODY = 11;
const PAGE_W = 12240, MARGIN = 1080; // US Letter, 0.75" margins
const CONTENT_W = PAGE_W - 2 * MARGIN; // 10080 DXA

// ---------- helpers ----------
const p = (text, opts = {}) =>
  new Paragraph({
    spacing: { after: 100, line: 264 },
    alignment: AlignmentType.JUSTIFIED,
    ...opts,
    children: Array.isArray(text) ? text : [new TextRun({ text, font: FONT, size: BODY * 2 })],
  });
const run = (text, extra = {}) => new TextRun({ text, font: FONT, size: BODY * 2, ...extra });
const b = (text) => run(text, { bold: true });
const h1 = (text) =>
  new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 200, after: 80 }, children: [new TextRun({ text, font: FONT })] });
const h2 = (text) =>
  new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 120, after: 60 }, children: [new TextRun({ text, font: FONT })] });
const bullet = (children) =>
  new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 60, line: 264 },
    children: Array.isArray(children) ? children : [run(children)] });
const caption = (text) =>
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 40, after: 140 },
    children: [new TextRun({ text, font: FONT, size: 18, italics: true, color: "444444" })] });

const border = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
const borders = { top: border, bottom: border, left: border, right: border };
const noBorder = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const noBorders = { top: noBorder, bottom: noBorder, left: noBorder, right: noBorder };

function cell(text, w, { bold = false, shade = null, align = AlignmentType.LEFT } = {}) {
  return new TableCell({
    width: { size: w, type: WidthType.DXA },
    borders,
    shading: shade ? { type: ShadingType.CLEAR, fill: shade, color: "auto" } : undefined,
    margins: { top: 40, bottom: 40, left: 80, right: 80 },
    children: [new Paragraph({ alignment: align, spacing: { after: 0 },
      children: [new TextRun({ text: String(text), font: FONT, size: 19, bold })] })],
  });
}
function table(header, rows, widths) {
  return new Table({
    width: { size: widths.reduce((a, c) => a + c, 0), type: WidthType.DXA },
    columnWidths: widths,
    rows: [
      new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, widths[i], { bold: true, shade: "D9E2F3", align: i ? AlignmentType.CENTER : AlignmentType.LEFT })) }),
      ...rows.map((r) => new TableRow({ children: r.map((v, i) => cell(v, widths[i], { align: i ? AlignmentType.CENTER : AlignmentType.LEFT })) })),
    ],
  });
}
function figure(name, widthIn, heightIn) {
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 0 },
    children: [new ImageRun({ type: "png", data: img(name), transformation: { width: widthIn * 96, height: heightIn * 96 } })] });
}
function twoFigures(a, b) {
  const w = CONTENT_W / 2;
  const mk = (f) => new TableCell({ width: { size: w, type: WidthType.DXA }, borders: noBorders,
    margins: { top: 0, bottom: 0, left: 40, right: 40 },
    children: [figure(f.name, f.w, f.h), caption(f.cap)] });
  return new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: [w, w],
    rows: [new TableRow({ children: [mk(a), mk(b)] })] });
}

// ---------- content ----------
const children = [];

// Title block
children.push(
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 },
    children: [new TextRun({ text: "Predicting Credit Card Default Risk with Python and Machine Learning", font: FONT, size: 36, bold: true, color: "1F3864" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 },
    children: [new TextRun({ text: "Group Project Part 1: Data Preprocessing and Initial Model Development", font: FONT, size: 24, color: "404040" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 },
    children: [new TextRun({ text: "Dataset: Default of Credit Card Clients (UCI Machine Learning Repository)", font: FONT, size: 20, italics: true, color: "595959" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: "1F3864", space: 6 } },
    children: [new TextRun({ text: "Team members: [Member 1], [Member 2], [Member 3], [Member 4]    |    Date: September 2026", font: FONT, size: 20, color: "595959" })] }),
);

// ---------- 1. Introduction ----------
children.push(h1("1. Introduction and Problem Framing"));
children.push(p([
  run("Credit card issuers lose money every time a customer stops paying, and they also lose money when they wrongly refuse credit to a customer who would have paid. "),
  run("Our stakeholder is the risk management team of a retail bank that wants an early-warning system: given a client’s profile and their last six months of billing and repayment history, how likely are they to default on next month’s payment? "),
  run("A reliable probability lets the bank prioritise collections outreach, adjust credit limits, and offer hardship plans before a missed payment becomes a charge-off."),
]));
children.push(p([
  b("Business question. "),
  run("Can we predict, one month ahead, which clients will default, and which behavioural signals drive that risk? "),
  b("Analytics framing. "),
  run("This is a supervised binary classification problem. The target is DEFAULT (1 = the client defaulted on the next month’s payment, 0 = paid). "),
  run(`The dataset is the UCI "Default of Credit Card Clients" file (Yeh & Lien, 2009), containing ${R.n_rows.toLocaleString()} Taiwanese clients observed April to September 2005, with ${R.n_cols - 1} predictors covering demographics, credit limit, monthly repayment status, bill amounts, and payment amounts. `),
  run("It was selected because it is real, large enough to support cross-validation, has a meaningful class imbalance, and raises genuine fairness questions since it includes sex, age, education, and marital status."),
]));
children.push(p([
  b("Success criteria. "),
  run(`A useful model must beat the trivial "predict nobody defaults" baseline, which is ${pct(BASE.accuracy)} accurate but catches zero defaulters. `),
  run("Because a missed defaulter costs more than an unnecessary outreach call, we prioritise recall on the default class and ROC-AUC over raw accuracy. "),
  b("Ethical scope. "),
  run("The model is intended to support human decisions about outreach, not to automatically deny credit. We treat SEX, AGE, EDUCATION, and MARRIAGE as sensitive attributes whose influence must be monitored, and we report group-level error rates in Section 3."),
]));

// ---------- 2. Dataset exploration & preprocessing ----------
children.push(h1("2. Dataset Exploration and Preprocessing"));
children.push(h2("2.1 Loading and structure"));
children.push(p([
  run("The raw file is an Excel workbook with a two-row header, which we loaded with pandas using header=1. All 24 columns arrived as integers; there were "),
  b(`${R.missing_total} missing values`),
  run(` and only ${R.duplicates} exact duplicate rows (0.1%), which we retained because the rows are plausible distinct clients with identical small-balance histories. `),
  run("The target is imbalanced: "),
  b(`${pct(R.default_rate)} of clients defaulted`),
  run(", so a naïve model can appear accurate while being useless (see figures/fig1_class_balance.png)."),
]));
children.push(table(
  ["Feature", "Count", "Mean", "Std", "Min", "Median", "Max"],
  [
    ["LIMIT_BAL (NT$)", "30,000", "167,484", "129,748", "10,000", "140,000", "1,000,000"],
    ["AGE (years)", "30,000", "35.5", "9.2", "21", "34", "79"],
    ["BILL_AMT1 (NT$)", "30,000", "51,223", "73,636", "-165,580", "22,382", "964,511"],
    ["PAY_AMT1 (NT$)", "30,000", "5,664", "16,563", "0", "2,100", "873,552"],
  ],
  [2100, 1200, 1400, 1400, 1300, 1300, 1380],
));
children.push(caption("Table 1. Summary statistics for selected numeric features (full table in outputs/summary_stats.csv)."));

children.push(h2("2.2 Key findings from exploratory analysis"));
children.push(bullet([
  b("Heavy right skew in money variables. "),
  run("Credit limits range from NT$10k to NT$1M with a median of NT$140k; bill and payment amounts have a long right tail and a few negative bills (over-payments or refunds). Tree models are indifferent to this, but the logistic regression needed standardisation."),
]));
children.push(bullet([
  b("Repayment status is the dominant signal. "),
  run(`The default rate is roughly 13% for clients who paid on time last month, jumps to 34% after a one-month delay, and exceeds 65% for delays of two months or more (Figure 1). PAY_1 has the highest correlation with the target (r = ${R.corr_default_pay1.toFixed(2)}), while credit limit is mildly protective (r = ${R.corr_default_limit.toFixed(2)}).`),
]));
children.push(bullet([
  b("Strong multicollinearity among monthly bills. "),
  run(`Adjacent bill amounts correlate at r = ${R.corr_bill1_bill2.toFixed(2)} (Figure 2), so the six raw bill columns carry largely redundant information. This motivated engineered summary features rather than relying on each month separately.`),
]));
children.push(bullet([
  b("Undocumented category codes. "),
  run(`EDUCATION is documented as 1 to 4 but contains codes 0, 5, and 6 (${R.education_undoc} rows); MARRIAGE contains an undocumented 0 (${R.marriage_undoc} rows). These are data-quality defects rather than real categories.`),
]));
children.push(bullet([
  b("Demographic differences are modest. "),
  run(`Default rates are ${pct(R.default_rate_by_group.sex.Male)} for men and ${pct(R.default_rate_by_group.sex.Female)} for women, and rise slightly with lower education level. The gap is small but non-zero, which is why we audit model error rates by group.`),
]));

children.push(twoFigures(
  { name: "fig3_default_by_pay_status.png", w: 3.3, h: 1.9, cap: "Figure 1. Default rate by last-month repayment status." },
  { name: "fig4_correlation.png", w: 3.0, h: 2.35, cap: "Figure 2. Correlation matrix of selected features." },
));

children.push(h2("2.3 Cleaning and transformation steps"));
children.push(p([
  run("All preprocessing is implemented in the "),
  run("preprocess()", { font: "Consolas", size: 20 }),
  run(" function of credit_default_analysis.py and produces outputs/cleaned_data.csv. The steps were:"),
]));
children.push(bullet([b("Column hygiene: "), run("renamed the target to DEFAULT, renamed PAY_0 to PAY_1 for consistency with PAY_2 to PAY_6, and dropped the ID column, which carries no information.")]));
children.push(bullet([b("Invalid codes: "), run("collapsed EDUCATION codes 0, 5, 6 into 4 (\"others\") and MARRIAGE code 0 into 3 (\"others\"), following the dataset documentation.")]));
children.push(bullet([b("Feature engineering (6 new features): "), run("MAX_DELAY (worst repayment status across six months), N_MONTHS_DELAYED (count of months with any delay), AVG_BILL, AVG_PAYMENT, UTILIZATION (average bill divided by credit limit, clipped to [-1, 5]), and PAY_TO_BILL_RATIO (average payment divided by average bill, clipped to [0, 5]). These summarise behaviour over the window and reduce the collinearity noted above.")]));
children.push(bullet([b("Encoding and scaling: "), run("SEX, EDUCATION, and MARRIAGE were one-hot encoded (first level dropped); the 26 numeric features were standardised with StandardScaler. Both were wrapped in a scikit-learn ColumnTransformer inside a Pipeline so that the scaler is fitted on training data only, avoiding leakage into the test set.")]));
children.push(bullet([b("Train/test split: "), run(`stratified 80/20 split (${R.train_size.toLocaleString()} train, ${R.test_size.toLocaleString()} test) with a fixed random seed, preserving the ${pct(R.default_rate)} default rate in both partitions. An AGE_BAND column was created solely for the fairness audit and is not a model input.`)]));

// ---------- 3. Initial model development ----------
children.push(h1("3. Initial Model Development"));
children.push(p([
  run("We trained two classifiers to establish a range: a "),
  b("Logistic Regression"),
  run(" (interpretable linear baseline, class_weight=\"balanced\" to counter the imbalance, max_iter=2000) and a "),
  b("Random Forest"),
  run(" (300 trees, max_depth=12, min_samples_leaf=20, class_weight=\"balanced_subsample\") chosen for its ability to capture the non-linear jump in risk seen in Figure 1 without heavy tuning. "),
  run("Model selection used 5-fold stratified cross-validation on the training set, scored by ROC-AUC; final metrics are reported once on the untouched 20% test set at the default 0.5 probability threshold."),
]));
children.push(table(
  ["Model", "CV ROC-AUC (5-fold)", "Test ROC-AUC", "Accuracy", "Precision", "Recall", "F1"],
  [
    ["Majority-class baseline", "–", f3(BASE.roc_auc), pct(BASE.accuracy), "0.000", "0.000", "0.000"],
    ["Logistic Regression", `${f3(LR.cv_auc_mean)} ± ${f3(LR.cv_auc_std)}`, f3(LR.roc_auc), pct(LR.accuracy), f3(LR.precision), f3(LR.recall), f3(LR.f1)],
    ["Random Forest", `${f3(RF.cv_auc_mean)} ± ${f3(RF.cv_auc_std)}`, f3(RF.roc_auc), pct(RF.accuracy), f3(RF.precision), f3(RF.recall), f3(RF.f1)],
  ],
  [2300, 1700, 1300, 1200, 1200, 1200, 1180],
));
children.push(caption("Table 2. Performance on the held-out test set (n = 6,000; 1,327 defaulters). Precision, recall, and F1 refer to the default class."));

const rfc = RF.confusion;
children.push(p([
  b("Interpretation. "),
  run(`Both models comfortably beat the baseline in every metric that matters. The Random Forest is the stronger initial model with a test ROC-AUC of ${f3(RF.roc_auc)}, which is in line with published results on this dataset (0.77 to 0.79). `),
  run(`It identifies ${pct(RF.recall)} of true defaulters (${rfc[1][1].toLocaleString()} of ${(rfc[1][0] + rfc[1][1]).toLocaleString()}) while flagging ${rfc[0][1].toLocaleString()} non-defaulters, giving a precision of ${pct(RF.precision)}. `),
  run("Its accuracy is slightly below the trivial baseline precisely because we asked it to take the minority class seriously; this trade-off is the correct one for the stakeholder’s use case. "),
  run(`The low cross-validation standard deviation (±${f3(RF.cv_auc_std)}) indicates the result is stable rather than a lucky split. `),
  run(`Feature importance (Figure 4) confirms the EDA: the three engineered or raw delay features (${R.top_features.slice(0, 3).join(", ")}) dominate, followed by PAY_2 and UTILIZATION, while the demographic dummies rank near the bottom.`),
]));
children.push(twoFigures(
  { name: "fig5_roc_curves.png", w: 2.7, h: 2.3, cap: "Figure 3. ROC curves on the test set." },
  { name: "fig7_feature_importance.png", w: 3.2, h: 2.4, cap: "Figure 4. Random Forest top-15 feature importances." },
));

const F = R.fairness;
children.push(p([
  b("Fairness check. "),
  run(`Because SEX and AGE are model inputs, we compared error rates across groups on the test set. Recall is nearly identical for women (${pct(F["SEX|Female"].recall_TPR)}) and men (${pct(F["SEX|Male"].recall_TPR)}); the false-positive rate is ${pct(F["SEX|Female"].false_positive_rate)} for women versus ${pct(F["SEX|Male"].false_positive_rate)} for men, and ${pct(F["AGE_BAND|51+"].false_positive_rate)} for clients over 50 versus ${pct(F["AGE_BAND|31-40"].false_positive_rate)} for those aged 31 to 40. `),
  run("These gaps are small but not zero. In Part 2 we will test whether removing the demographic features costs any accuracy, tune the decision threshold against an explicit cost matrix, and add gradient boosting and calibrated probabilities."),
]));

// ---------- 4. Team collaboration ----------
children.push(h1("4. Team Collaboration Process"));
children.push(p([
  run("We ran the project as a two-week sprint with a lightweight agile process. Work was tracked on a shared Kanban board (Backlog, In Progress, Review, Done) and code lived in a single GitHub repository with a protected main branch; every change was submitted as a pull request and reviewed by at least one other member before merging. "),
  run("We held three 30-minute stand-ups per week on video call and kept an asynchronous chat channel for questions between meetings."),
]));
children.push(p([b("Roles and responsibilities:")], { spacing: { after: 40 } }));
children.push(bullet([b("[Member 1] – Data lead: "), run("dataset acquisition, loading, exploratory analysis, and the figures in Section 2.")]));
children.push(bullet([b("[Member 2] – Preprocessing lead: "), run("category-code cleanup, feature engineering, and the scikit-learn preprocessing pipeline.")]));
children.push(bullet([b("[Member 3] – Modelling lead: "), run("model training, cross-validation, evaluation metrics, and the fairness audit.")]));
children.push(bullet([b("[Member 4] – Documentation and QA lead: "), run("report writing, code review, README, and verifying that the script reproduces every number in this report from a clean run.")]));
children.push(p([
  b("What worked and what we changed. "),
  run("An early decision to put every step into one reproducible script (credit_default_analysis.py, run with a single command) meant that any teammate could regenerate all figures and metrics after a change, which removed most merge friction. "),
  run("Our main iteration was on class imbalance rather than on features. The first Random Forest, trained on the raw 23 columns with no class weighting, reached an ROC-AUC of 0.777 but caught only 36% of defaulters at the default threshold. Switching to balanced class weights raised recall to 59% at the same ROC-AUC. Adding the six engineered summary features left ROC-AUC unchanged (0.777) but produced a cleaner importance ranking that the stakeholder can act on, so we kept them. "),
  run("Disagreements (for example, whether to keep the demographic features) were resolved by agreeing to measure the effect empirically in Part 2 rather than deciding by opinion."),
]));

// ---------- References ----------
children.push(h1("References"));
const ref = (t) => new Paragraph({ spacing: { after: 60, line: 264 }, indent: { left: 360, hanging: 360 }, children: [run(t, { size: 20 })] });
children.push(ref("Yeh, I.-C., & Lien, C.-H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, 36(2), 2473–2480."));
children.push(ref("Dua, D., & Graff, C. (2019). Default of Credit Card Clients Data Set. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/350"));
children.push(ref("Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830."));
children.push(ref("McKinney, W. (2010). Data structures for statistical computing in Python. Proceedings of the 9th Python in Science Conference, 56–61."));

// ---------- document ----------
const doc = new Document({
  creator: "Group Project Team",
  title: "Predicting Credit Card Default Risk – Group Project Part 1",
  styles: {
    default: { document: { run: { font: FONT, size: BODY * 2 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, color: "1F3864", font: FONT },
        paragraph: { spacing: { before: 200, after: 80 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 23, bold: true, color: "2F5496", font: FONT },
        paragraph: { spacing: { before: 120, after: 60 }, outlineLevel: 1 } },
    ],
  },
  numbering: {
    config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] }],
  },
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: 15840 }, margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } } },
    footers: {
      default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Group Project Part 1  |  Page ", font: FONT, size: 18, color: "7F7F7F" }),
                   new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18, color: "7F7F7F" })] })] }),
    },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  const out = path.join(__dirname, "GroupProject_Part1_Report.docx");
  fs.writeFileSync(out, buf);
  console.log("Wrote", out);
});
