"""CityTransit Control: portable ETA regression prototype (standard library only)."""
from pathlib import Path
import math, random, json, csv, statistics, time
ROOT=Path(__file__).resolve().parents[1]
FEATURES=['bias','base_minutes','base_x_traffic','occupancy_ratio','rain','peak']

def features(r):
    base=r['distance_remaining_km']/max(r['speed_kmh'],5)*60
    return [1,base,base*r['traffic_level'],r['occupancy_ratio'],int(r['weather']=='rain'),int(7<=r['hour']<=9 or 16<=r['hour']<=19)]

def solve(a,b):
    a=[list(row)+[value] for row,value in zip(a,b)]
    for i in range(len(b)):
        pivot=max(range(i,len(b)),key=lambda k:abs(a[k][i]))
        a[i],a[pivot]=a[pivot],a[i]
        if abs(a[i][i])<1e-12: raise ValueError('singular matrix')
        div=a[i][i];a[i]=[x/div for x in a[i]]
        for j in range(len(b)):
            if j!=i:
                div=a[j][i];a[j]=[x-div*y for x,y in zip(a[j],a[i])]
    return [row[-1] for row in a]

def fit(rows):
    x=[features(r) for r in rows]; n=len(FEATURES)
    a=[[sum(row[i]*row[j] for row in x)+(0.01 if i==j and i else 0) for j in range(n)] for i in range(n)]
    b=[sum(row[i]*r['actual_eta_min'] for row,r in zip(x,rows)) for i in range(n)]
    return solve(a,b)

def estimate(r,coef):return max(0,sum(a*b for a,b in zip(features(r),coef)))

def validate(r):
    needed=['distance_remaining_km','speed_kmh','traffic_level','occupancy_ratio','weather','hour','telemetry_age_sec']
    if any(r.get(k) is None for k in needed): return 'missing_field'
    for k in needed:
        if k!='weather' and (not isinstance(r[k],(int,float)) or not math.isfinite(r[k])):return 'invalid_type'
    if not (0<=r['telemetry_age_sec']<=60):return 'stale_telemetry'
    for k,lo,hi in [('distance_remaining_km',.2,6),('speed_kmh',5,60),('traffic_level',1,5),('occupancy_ratio',0,1),('hour',0,23)]:
        if not lo<=r[k]<=hi:return 'out_of_domain_'+k
    if r['weather'] not in ['dry','rain']:return 'unknown_weather'
    return None

def predict(r,model,available=True):
    err=validate(r)
    if err or not available:
        return {'eta_min':None,'interval':None,'status':'reject','reason':err or 'model_unavailable','fallback':'schedule_labelled','model_version':model['version']}
    eta=estimate(r,model['coefficients']);q=model['q90']
    # Experimental half-width gate; no operational command is ever automated.
    status='review' if q>2 or r['traffic_level']==5 or r['occupancy_ratio']>=.95 else 'auto_information'
    return {'eta_min':round(eta,2),'interval':[round(max(0,eta-q),2),round(eta+q,2)],'status':status,'reason':'adverse_context' if status=='review' else 'fresh_in_domain','fallback':None,'model_version':model['version']}

def synthetic(n,seed,prefix):
    rng=random.Random(seed);out=[]
    for i in range(n):
        r={'record_id':prefix+str(i),'route_id':'R-'+str(rng.choice([8,27,31])),'distance_remaining_km':round(rng.uniform(.2,6),3),'speed_kmh':round(rng.uniform(5,60),2),'traffic_level':rng.randint(1,5),'occupancy_ratio':round(rng.random(),3),'weather':rng.choice(['dry','rain']),'hour':rng.randint(0,23),'telemetry_age_sec':rng.randint(0,45)}
        x=features(r)
        r['actual_eta_min']=round(max(.1,1+.85*x[1]+.08*x[2]+.8*x[3]+.7*x[4]+.8*x[5]+rng.gauss(0,.65)),3)
        out.append(r)
    return out

def metrics(rows,coef,q):
    errors=[estimate(r,coef)-r['actual_eta_min'] for r in rows]
    baseline=[r['distance_remaining_km']/max(r['speed_kmh'],5)*60-r['actual_eta_min'] for r in rows]
    return {'n':len(rows),'mae_min':statistics.mean(abs(x) for x in errors),'rmse_min':math.sqrt(statistics.mean(x*x for x in errors)),'within_2min':sum(abs(x)<=2 for x in errors)/len(errors),'interval_coverage':sum(abs(x)<=q for x in errors)/len(errors),'baseline_mae_min':statistics.mean(abs(x) for x in baseline)}

def write_csv(path,rows):
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    train=synthetic(1200,301,'TR-');cal=synthetic(200,302,'CA-');test=synthetic(200,303,'TE-')
    coef=fit(train);res=sorted(abs(estimate(r,coef)-r['actual_eta_min']) for r in cal)
    q=res[min(len(res)-1,math.ceil((len(cal)+1)*.9)-1)]
    model={'version':'eta-ridge-v1','features':FEATURES,'coefficients':coef,'q90':q,'train_n':len(train),'calibration_n':len(cal),'provenance':'Synthetic educational data; not calibrated for real traffic'}
    (ROOT/'results/model.json').write_text(json.dumps(model,indent=2))
    for name,rows in [('train',train),('calibration',cal),('test',test)]:write_csv(ROOT/('data/'+name+'.csv'),rows)
    base={'route_id':'R-31','distance_remaining_km':2,'speed_kmh':25,'traffic_level':3,'occupancy_ratio':.6,'weather':'dry','hour':12,'telemetry_age_sec':10}
    specs=[('normal',{},7.43),('light_traffic',{'traffic_level':1,'occupancy_ratio':.2,'speed_kmh':40},3.95),('rush_rain',{'hour':8,'weather':'rain','traffic_level':4},9.14),('congested',{'traffic_level':5,'speed_kmh':10},17.48),('near_stop',{'distance_remaining_km':.3,'speed_kmh':18},2.57),('long_trip',{'distance_remaining_km':5.5,'speed_kmh':35},11.76),('crowded',{'occupancy_ratio':.98},7.72),('night',{'hour':2,'traffic_level':1},5.94),('rain',{'weather':'rain'},8.25),('road_closure',{},22.0),('snow',{'weather':'snow'},14.0),('stale',{'telemetry_age_sec':180},8.0),('zero_speed',{'speed_kmh':0},18.0),('missing',{'distance_remaining_km':None},8.0),('unavailable',{},8.0)]
    cases=[]
    for name,patch,actual in specs:
        r=dict(base,**patch);r['case_id']=name
        p=predict(r,model,available=name!='unavailable')
        expected='reject' if name in ['snow','stale','zero_speed','missing','unavailable'] else ('review' if name in ['congested','crowded'] else 'auto_information')
        err=None if p['eta_min'] is None else round(abs(p['eta_min']-actual),2)
        cases.append({'case':name,'input':r,'actual_eta_min':actual,'expected_status':expected,'output':p,'abs_error_min':err,'policy_ok':p['status']==expected,'within_2min':None if err is None else err<=2})
    m=metrics(test,coef,q);m['scenario_numeric_n']=sum(c['abs_error_min'] is not None for c in cases);m['scenario_within2_n']=sum(c['within_2min'] is True for c in cases);m['policy_passed']=sum(c['policy_ok'] for c in cases);m['policy_total']=len(cases)
    (ROOT/'results/metrics.json').write_text(json.dumps(m,indent=2));(ROOT/'results/scenarios.json').write_text(json.dumps(cases,indent=2))
    checks=[]
    checks.append(('split_ids_disjoint',not(set(r['record_id'] for r in train)&set(r['record_id'] for r in test))))
    checks.append(('all_policy_cases',all(c['policy_ok'] for c in cases)))
    checks.append(('missing_no_prediction',predict(dict(base,distance_remaining_km=None),model)['eta_min'] is None))
    checks.append(('nan_rejected',predict(dict(base,speed_kmh=float('nan')),model)['status']=='reject'))
    checks.append(('age_60_valid',predict(dict(base,telemetry_age_sec=60),model)['status']!='reject'))
    checks.append(('age_61_rejected',predict(dict(base,telemetry_age_sec=61),model)['status']=='reject'))
    checks.append(('no_driver_command',all('command' not in c['output'] for c in cases)))
    (ROOT/'results/checks.json').write_text(json.dumps(checks,indent=2))
    if not all(v for _,v in checks):raise AssertionError(checks)
    print(json.dumps({'metrics':m,'model':model,'checks':checks},indent=2))
    return model,cases,m
if __name__=='__main__':main()
