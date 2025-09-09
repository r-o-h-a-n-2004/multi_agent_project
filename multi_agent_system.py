import json
import asyncio
from typing import List, Dict, Any, Optional, TypedDict
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from ddgs import DDGS

from config import Config

# Initialize LLM
llm = ChatOpenAI(
    model=Config.MODEL_NAME,
    temperature=Config.MODEL_TEMPERATURE
)

# Define the State for LangGraph (as a TypedDict)
class AgentState(TypedDict):
    company_name: str
    research_findings: Optional[str]
    industry: Optional[str]
    key_offerings: Optional[List[str]]
    strategic_focus: Optional[List[str]]
    use_cases: Optional[List[Dict[str, Any]]]
    resources: Optional[List[Dict[str, Any]]]
    final_report: Optional[str]
    error: Optional[str]

# DuckDuckGo search function (unchanged)
def search_duckduckgo(query: str, max_results: int = Config.MAX_SEARCH_RESULTS):
    """Search using DuckDuckGo without API keys."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            return results
    except Exception as e:
        return [{"title": "Search Error", "body": f"DuckDuckGo search failed: {str(e)}"}]

# Custom search tools (unchanged)
def search_web(query: str):
    """Search for general company/industry information."""
    results = search_duckduckgo(query)
    formatted_results = []
    for result in results:
        formatted_results.append(f"Title: {result.get('title', 'N/A')}\nSnippet: {result.get('body', 'N/A')}\nURL: {result.get('href', 'N/A')}\n")
    return "\n".join(formatted_results)

def search_datasets(query: str):
    """Search for AI/ML datasets on specific platforms."""
    platform_queries = [
        f"{query} dataset site:kaggle.com",
        f"{query} dataset site:huggingface.co",
        f"{query} dataset site:github.com",
        f"{query} AI ML dataset"
    ]
    
    all_results = []
    for search_query in platform_queries:
        results = search_duckduckgo(search_query, max_results=3)
        all_results.extend(results)
    
    formatted_results = []
    for result in all_results:
        formatted_results.append(f"Title: {result.get('title', 'N/A')}\nSnippet: {result.get('body', 'N/A')}\nURL: {result.get('href', 'N/A')}\n")
    
    return "\n".join(formatted_results)

# Define agent nodes for the graph
async def research_agent(state: AgentState):
    """Research the company/industry and extract key information."""
    try:
        print(f"🔍 Researching {state['company_name']}...")
        
        search_query = f"{state['company_name']} company industry products services business model"
        search_results = search_web(search_query)
        
        research_prompt = f"""
        You are an expert AI business analyst. Analyze the following information about the company "{state['company_name']}" and extract structured information.
        Requirements:
        1. Identify the **industry and market segment** (e.g., Automotive, Retail, Healthcare, etc.).
        2. List **key products/services/offerings** (3-5 main items).
        3. List **strategic focus areas** (operations, supply chain, customer experience, marketing, etc.).
        4. List **current challenges or opportunities** mentioned.
        
        Search Results:
        {search_results}

        ⚠ IMPORTANT INSTRUCTIONS:
        - Return your answer **ONLY in strict JSON format**.
        - Do NOT include any explanation, extra text, or commentary.
        - Follow the JSON structure exactly as shown in the example.
        - Use "Unable to determine" if a value cannot be inferred.
        
        Example JSON output:
        {{
            "industry": "Retail & E-commerce",
            "key_offerings": ["Sportswear", "Athletic Shoes", "Accessories"],
            "strategic_focus": ["Digital Transformation", "Supply Chain Optimization", "Customer Experience"],
            "challenges": ["Competition from direct-to-consumer brands", "Supply chain disruptions"]
        }}
        """
        
        response = llm.invoke([
            HumanMessage(content=research_prompt)
        ])
        
        try:
            research_data = json.loads(response.content)
            return {
                "research_findings": search_results,
                "industry": research_data.get("industry", "Unable to determine"),
                "key_offerings": research_data.get("key_offerings", ["Unable to determine"]),
                "strategic_focus": research_data.get("strategic_focus", ["Unable to determine"])
            }
        except json.JSONDecodeError:
            return {
                "research_findings": search_results,
                "industry": "Unable to determine",
                "key_offerings": ["Unable to determine"],
                "strategic_focus": ["Unable to determine"]
            }
            
    except Exception as e:
        return {"error": f"Research failed: {str(e)}"}

async def use_case_agent(state: AgentState):
    """Generate relevant AI/GenAI use cases based on research."""
    try:
        print("💡 Generating use cases...")
        
        industry_trends_query = f"AI GenAI ML trends in {state['industry']} industry 2024"
        trends_results = search_web(industry_trends_query)
        
        use_case_prompt = f"""
        Based on the following company research and industry trends, generate 5-7 relevant AI/GenAI use cases for {state['company_name']}:
        
        COMPANY ANALYSIS:
        Industry: {state['industry']}
        Key Offerings: {', '.join(state['key_offerings']) if state['key_offerings'] else 'N/A'}
        Strategic Focus: {', '.join(state['strategic_focus']) if state['strategic_focus'] else 'N/A'}
        
        INDUSTRY AI TRENDS:
        {trends_results}
        
        Generate use cases that consider:
        1. Large Language Models (LLMs) applications
        2. Generative AI solutions
        3. Machine Learning automation
        4. Operational efficiency improvements
        5. Customer experience enhancement
        
        For each use case, provide:
        - Title
        - Description (2-3 sentences)
        - Potential impact/business value
        - Key technologies required (LLM, Computer Vision, NLP, etc.)
        
        Return in JSON format with a list of use cases under the key "use_cases".
        """
        
        response = llm.invoke([
            HumanMessage(content=use_case_prompt)
        ])
        
        try:
            use_cases_data = json.loads(response.content)
            return {"use_cases": use_cases_data.get("use_cases", [])}
        except json.JSONDecodeError:
            fallback_cases = [
                {
                    "title": "AI-Powered Customer Support",
                    "description": "Implement LLM-based chatbot for handling customer queries and support tickets",
                    "impact": "Reduce support costs by 40%, improve response time, 24/7 availability",
                    "technologies": ["LLM", "NLP", "Chatbot Framework"]
                },
                {
                    "title": "Personalized Product Recommendations",
                    "description": "ML-based recommendation engine for personalized shopping experiences",
                    "impact": "Increase conversion rates by 25%, improve customer engagement",
                    "technologies": ["Machine Learning", "Recommendation Algorithms"]
                }
            ]
            return {"use_cases": fallback_cases}
            
    except Exception as e:
        return {"error": f"Use case generation failed: {str(e)}"}

async def resource_agent(state: AgentState):
    """Find relevant resources for the generated use cases."""
    try:
        print("📚 Collecting resources...")
        
        if not state['use_cases']:
            return {"resources": []}
        
        resources = []
        
        for use_case in state['use_cases'][:3]:
            title = use_case.get('title', '')
            
            dataset_query = f"{title} {state['industry']} dataset"
            dataset_results = search_datasets(dataset_query)
            
            implementation_query = f"{title} implementation guide tutorial best practices"
            implementation_results = search_web(implementation_query)
            
            resources.append({
                "use_case": title,
                "datasets": dataset_results,
                "implementation_guides": implementation_results
            })
        
        return {"resources": resources}
        
    except Exception as e:
        return {"error": f"Resource collection failed: {str(e)}"}

async def report_agent(state: AgentState):
    """Generate the final comprehensive report."""
    try:
        print("📊 Generating final report...")
        
        report_prompt = f"""
        Create a comprehensive AI consulting report for {state['company_name']}.
        
        COMPANY ANALYSIS:
        Industry: {state['industry']}
        Key Offerings: {', '.join(state['key_offerings']) if state['key_offerings'] else 'N/A'}
        Strategic Focus Areas: {', '.join(state['strategic_focus']) if state['strategic_focus'] else 'N/A'}
        
        PROPOSED AI USE CASES:
        {json.dumps(state['use_cases'], indent=2) if state['use_cases'] else 'No use cases generated'}
        
        RESOURCES:
        {json.dumps(state['resources'], indent=2) if state['resources'] else 'No resources found'}
        
        Format the report as a professional markdown document with the following sections:
        
        # AI Consultation Report for {state['company_name']}
        
        ## Executive Summary
        Brief overview of findings and recommendations
        
        ## Company & Industry Analysis
        - Industry: {state['industry']}
        - Key Offerings
        - Strategic Focus Areas
        
        ## AI Trends in {state['industry']}
        Current industry trends and opportunities
        
        ## Proposed AI/GenAI Use Cases
        Detailed description of each use case with:
        - Title
        - Description
        - Expected Impact
        - Required Technologies
        - Implementation Complexity
        
        ## Implementation Resources
        - Available datasets
        - Learning resources
        - Implementation guides
        
        ## Next Steps & Recommendations
        Actionable steps for implementation
        
        Make sure the report is professional, actionable, and includes clickable links where available.
        """
        
        response = llm.invoke([
            HumanMessage(content=report_prompt)
        ])
        
        return {"final_report": response.content}
        
    except Exception as e:
        return {"error": f"Report generation failed: {str(e)}"}

# Build the LangGraph workflow
def create_workflow():
    """Create and compile the LangGraph workflow."""
    workflow = StateGraph(AgentState)
    
    # Add nodes (agents)
    workflow.add_node("research_agent", research_agent)
    workflow.add_node("use_case_agent", use_case_agent)
    workflow.add_node("resource_agent", resource_agent)
    workflow.add_node("report_agent", report_agent)
    
    # Define the flow
    workflow.set_entry_point("research_agent")
    workflow.add_edge("research_agent", "use_case_agent")
    workflow.add_edge("use_case_agent", "resource_agent")
    workflow.add_edge("resource_agent", "report_agent")
    workflow.add_edge("report_agent", END)
    
    # Compile the graph
    return workflow.compile()

# Create the graph instance
graph = create_workflow()

# Main execution function
async def run_consultation(company_name: str):
    """Run the complete consultation workflow using LangGraph."""
    print(f"🚀 Starting AI consultation for: {company_name}")
    print("=" * 50)
    
    # Initialize state
    initial_state = AgentState(
        company_name=company_name,
        research_findings=None,
        industry=None,
        key_offerings=None,
        strategic_focus=None,
        use_cases=None,
        resources=None,
        final_report=None,
        error=None
    )
    
    # Execute the graph
    final_state = await graph.ainvoke(initial_state)
    
    print("=" * 50)
    print("✅ Consultation complete!")
    
    return final_state

# Display results (unchanged)
def display_results(result: AgentState):
    """Display the results in a formatted way."""
    print("\n" + "=" * 60)
    print("🎯 CONSULTATION RESULTS")
    print("=" * 60)
    
    print(f"\n📋 Company: {result['company_name']}")
    print(f"🏭 Industry: {result['industry']}")
    
    print(f"\n🎯 Key Offerings:")
    for offering in (result['key_offerings'] or []):
        print(f"   • {offering}")
    
    print(f"\n🎯 Strategic Focus Areas:")
    for focus in (result['strategic_focus'] or []):
        print(f"   • {focus}")
    
    print(f"\n🤖 Generated Use Cases ({len(result['use_cases'] or [])}):")
    for i, use_case in enumerate(result['use_cases'] or [], 1):
        print(f"\n   {i}. {use_case.get('title', 'N/A')}")
        print(f"      Description: {use_case.get('description', 'N/A')}")
        print(f"      Impact: {use_case.get('impact', 'N/A')}")
        print(f"      Technologies: {', '.join(use_case.get('technologies', []))}")
    
    print(f"\n📊 Final Report generated ({len(result['final_report'] or '')} characters)")
    print("\n" + "=" * 60)

# Save results to file (unchanged)
def save_results(result: AgentState, filename: str = "consultation_report.md"):
    """Save the consultation results to a markdown file."""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"# AI Consultation Report for {result['company_name']}\n\n")
        f.write(result['final_report'] or "No report generated")
    print(f"💾 Report saved to {filename}")

# Main function
async def main():
    """Main function to demonstrate the system."""
    try:
        test_companies = ["Nike", "Tesla", "Amazon"]  # Test with multiple companies
        
        for company_name in test_companies:
            print(f"\n{'='*60}")
            print(f"ANALYZING: {company_name}")
            print(f"{'='*60}")
            
            # Run the consultation
            result = await run_consultation(company_name)
            
            # Display results
            display_results(result)
            
            # Save to file
            save_results(result, f"consultation_report_{company_name.lower()}.md")
            
            # Print a sample of the report
            if result['final_report']:
                print("\n📄 Report Preview:")
                preview = result['final_report'][:500] + "..." if len(result['final_report']) > 500 else result['final_report']
                print(preview)
                
            print(f"\n{'='*60}")
            print(f"COMPLETED: {company_name}")
            print(f"{'='*60}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())