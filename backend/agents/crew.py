"""
LLM orchestration for DataMind AI.
Uses a direct LangChain call to generate a comprehensive analysis report.
"""
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from config import settings


def get_llm():
    """Return the configured LLM instance."""
    if settings.LLM_PROVIDER == "openai" and settings.OPENAI_API_KEY:
        api_key = settings.OPENAI_API_KEY.strip("\"' ")
        return ChatOpenAI(model=settings.LLM_MODEL, temperature=0.2, api_key=api_key)
    elif settings.LLM_PROVIDER == "groq" and settings.GROQ_API_KEY:
        api_key = settings.GROQ_API_KEY.strip("\"' ")
        return ChatGroq(model=settings.LLM_MODEL, temperature=0.2, api_key=api_key)
    elif settings.LLM_PROVIDER == "anthropic" and settings.ANTHROPIC_API_KEY:
        api_key = settings.ANTHROPIC_API_KEY.strip("\"' ")
        return ChatAnthropic(model=settings.LLM_MODEL, temperature=0.2, api_key=api_key)
    return None


REPORT_SYSTEM_PROMPT = """You are DataMind AI, an expert data analyst and report writer.
Your job is to produce a comprehensive, professional markdown analysis report from the provided dataset information.

The report MUST include these sections (use ## headings):
1. **Executive Summary** – 3-4 sentence overview of the dataset and key findings
2. **Dataset Overview** – shape, columns, data types, memory usage
3. **Data Quality Assessment** – missing values, duplicates, outliers found and how they were handled
4. **Exploratory Data Analysis** – key statistics, distributions, correlations, notable patterns
5. **Key Insights** – top 5 actionable findings with bullet points
6. **Machine Learning Potential** – recommended models and expected performance for this data type
7. **Recommendations** – concrete next steps for the data owner
8. **Conclusion** – summary paragraph

Use markdown formatting: tables, bullet lists, bold text. Be specific with numbers from the data.
Do NOT make up numbers that aren't in the provided analysis – only use what is given."""


class AnalysisCrew:
    """
    Generates a comprehensive AI analysis report using a single powerful LLM call.
    Replaces the previous multi-agent CrewAI pipeline for reliability and speed.
    """

    def __init__(self, dataset_info: str, analysis_type: str = "full"):
        self.dataset_info = dataset_info
        self.analysis_type = analysis_type
        self.llm = get_llm()

    def run_full_analysis(self, cleaning_report: str, eda_results: str, ml_results: str = "") -> str:
        """Generate a comprehensive analysis report from cleaning + EDA results."""
        if self.llm is None:
            return self._fallback_report(cleaning_report, eda_results)

        user_prompt = f"""Please generate a full analysis report for this dataset.

## Dataset Information
{self.dataset_info}

## Data Cleaning Report
{cleaning_report}

## Exploratory Data Analysis Results
{eda_results}

## ML Results
{ml_results if ml_results else "Not run for this analysis."}

Generate the complete markdown report now."""

        try:
            response = self.llm.invoke([
                SystemMessage(content=REPORT_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ])
            return response.content
        except Exception as e:
            return self._fallback_report(cleaning_report, eda_results, error=str(e))

    def _fallback_report(self, cleaning_report: str, eda_results: str, error: str = "") -> str:
        """Return a structured markdown report built from raw tool outputs when LLM is unavailable."""
        import json
        lines = ["# 📊 DataMind Analysis Report\n"]
        lines.append(f"## Dataset Overview\n{self.dataset_info}\n")

        # Cleaning section
        try:
            cr = json.loads(cleaning_report) if isinstance(cleaning_report, str) else cleaning_report
            lines.append("## Data Cleaning Summary")
            if isinstance(cr, dict):
                for k, v in cr.items():
                    lines.append(f"- **{k}**: {v}")
            else:
                lines.append(str(cr))
            lines.append("")
        except Exception:
            lines.append(f"## Data Cleaning\n{cleaning_report}\n")

        # EDA section
        try:
            er = json.loads(eda_results) if isinstance(eda_results, str) else eda_results
            lines.append("## Exploratory Data Analysis")
            if isinstance(er, dict):
                for k, v in er.items():
                    if isinstance(v, dict):
                        lines.append(f"\n### {k}")
                        for sk, sv in list(v.items())[:10]:
                            lines.append(f"- **{sk}**: {sv}")
                    else:
                        lines.append(f"- **{k}**: {v}")
            else:
                lines.append(str(er))
            lines.append("")
        except Exception:
            lines.append(f"## EDA Results\n{eda_results}\n")

        if error:
            lines.append(f"\n> ⚠️ AI commentary unavailable: {error}")
            lines.append("> The statistical results above were computed directly from your data.")

        lines.append("\n---\n*Generated by DataMind AI*")
        return "\n".join(lines)
