from langgraph.graph import StateGraph, END
from .registry import get_tool
from .router import route
from .state import AgentState

def route_node(s):
    task,tools=route(s['query'],len(s.get('image_paths',[])))
    s['task']=task; s['selected_tools']=tools
    s.setdefault('trace',[]).append({'stage':'routing','task':task,'tools':tools})
    return s

def exec_node(s):
    results=[]
    for name in s.get('selected_tools',[]):
        try:
            r=get_tool(name)(s); results.append(r)
            s.setdefault('trace',[]).append({'stage':'tool_execution','tool':name,'model':r.get('model'),'confidence':r.get('confidence'),'status':'success'})
        except Exception as e:
            s.setdefault('trace',[]).append({'stage':'tool_execution','tool':name,'status':'error','error':f'{type(e).__name__}: {e}'})
            if not results: raise
    s['tool_results']=results; return s

def finalize_node(s):
    rs=s.get('tool_results',[])
    if not rs: s['answer']='No result.'; s['confidence']=0.0; return s
    primary=rs[-1]; s['answer']=primary.get('answer','Analysis complete.')
    s['evidence']=[e for r in rs for e in r.get('evidence',[])]
    s['confidence']=sum(float(r.get('confidence',0)) for r in rs)/len(rs)
    s.setdefault('trace',[]).append({'stage':'finalize','confidence':s['confidence'],'evidence_count':len(s['evidence'])})
    return s

g=StateGraph(AgentState); g.add_node('route',route_node); g.add_node('execute',exec_node); g.add_node('finalize',finalize_node); g.set_entry_point('route'); g.add_edge('route','execute'); g.add_edge('execute','finalize'); g.add_edge('finalize',END)
GRAPH=g.compile()
def run_agent(state): return GRAPH.invoke(state)
