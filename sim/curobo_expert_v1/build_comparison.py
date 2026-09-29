"""Create an evidence-linked local report; partial attempts stay visibly partial."""
import argparse
import html
import json
import os
from pathlib import Path

from .quality import sha256_file,validate_episode_provenance,verify_hashes
from .verify_media import validate_review_sources

ROOT=Path(__file__).resolve().parent


def relative(path):
    return os.path.relpath(Path(path).resolve(),ROOT)


def collect(spec):
    if not spec.get('episode_dir'):raise ValueError('explicit episode_dir required')
    episode=(ROOT/spec['episode_dir']).resolve()
    provenance=validate_episode_provenance(episode)
    if provenance.get('status')!='passed':raise ValueError('episode provenance rejected: '+str(provenance.get('reasons')))
    directory=(ROOT/spec['report_dir']).resolve()
    report_hashes={name:sha256_file(directory/name) for name in ('quality_report.json','free_motion_report.json')}
    quality=json.loads((directory/'quality_report.json').read_text())
    free=json.loads((directory/'free_motion_report.json').read_text())
    for bindings in (quality.get('artifact_sha256'),free.get('provenance',{}).get('artifact_sha256')):
        if not isinstance(bindings,dict):raise ValueError('report lacks artifact hash anchors')
        if any(bindings.get(name)!=digest for name,digest in provenance['artifact_sha256'].items()):
            raise ValueError('report artifact anchors differ from actual episode: '+spec['label'])
        failures=verify_hashes(episode,bindings)
        if failures:raise ValueError('report evidence changed: '+str(failures))
    outcome=json.loads((episode/'outcome.json').read_text())
    measured=quality.get('motion',{}).get('global',{}).get('measured')
    contact=quality.get('contact',{}).get('legacy',{}).get('points',[])
    if not measured or not contact:raise ValueError('candidate lacks measured motion/contact: '+spec['label'])
    controller_points=outcome.get('points',[])
    controller_names=[p.get('target') for p in controller_points]
    contact_names=[p.get('target') for p in contact]
    successful_names=[p.get('target') for p in controller_points if p.get('success') is True]
    if (not controller_names or any(not isinstance(name,str) for name in controller_names+contact_names)
            or len(set(controller_names))!=len(controller_names) or len(set(contact_names))!=len(contact_names)
            or contact_names!=[name for name in controller_names if name in set(contact_names)]
            or not set(successful_names)<=set(contact_names)):
        raise ValueError('controller/contact target inventory mismatch')
    names=[s+str(i) for i in range(1,6) for s in ('R','L')]
    completed=(outcome.get('success') is True and outcome.get('requested_points')==names
        and [p.get('target') for p in outcome.get('points',[])]==names
        and all(p.get('success') is True for p in outcome['points'])
        and [p['target'] for p in contact]==names and all(p.get('accepted') is True for p in contact)
        and quality['contact']['legacy'].get('accepted') is True
        and all(provenance.get('recorded_acceptance',{}).get(name) is True for name in ('independent_audit.json','posture_audit.json')))
    result=dict(label=spec['label'],description=spec['description'],episode=relative(episode),
        quality=relative(directory/'quality_report.json'),free=relative(directory/'free_motion_report.json'),
        quality_sha256=report_hashes['quality_report.json'],
        free_sha256=report_hashes['free_motion_report.json'],
        complete=completed,
        completed_points=len(successful_names),controller_attempted_points=len(controller_points),
        completed_points_scope='bound controller outcome point.success; not whole-episode independent audit acceptance',
        audit_accepted_points=sum(p.get('accepted') is True for p in contact),recorded_points=len(contact),
        hard_failures=quality['hard_failures'],decision=quality['simulation_quality_decision'],
        free_status=free['status'],free_failures=free['failures'],
        travel=measured['total_travel_rad'],per_joint=measured['travel_rad'],reversals=measured['reversal_count'],
        duration=quality['motion']['sampling']['duration_s'],
        elbow_mm=1000*measured['posture']['max_elbow_above_shoulder_m'],
        force_peak=max(p['max_physical_force_N'] for p in contact),
        hold_min=min(p['continuous_front_contact_hold_s'] for p in contact),
        acceleration_raw=max(measured['derivatives']['raw']['acceleration']['max_abs_per_joint']),
        jerk_raw=max(measured['derivatives']['raw']['jerk']['max_abs_per_joint']),
        jerk_filtered=max(measured['derivatives']['filtered']['jerk']['max_abs_per_joint']),
        free_jerk=free['actual_windows']['jerk_rad_s3']['peak']['value'],clips={})
    if spec.get('review_dir'):
        manifest=(ROOT/spec['review_dir']/'review_manifest.json').resolve()
        review,clips=validate_review_sources(manifest)
        receipt=(ROOT/spec['media_verification']).resolve() if spec.get('media_verification') else manifest.parent/'media_verification.json'
        checked=json.loads(receipt.read_text())
        if (checked['source_bindings_passed'] is not True or checked['manifest_sha256']!=sha256_file(manifest)
                or checked.get('semantic_binding',{}).get('status')!='passed'
                or checked['semantic_binding'].get('frames_verified')!=sum(x['clip']['frame_count'] for x in clips)
                or checked.get('verifier_sha256')!=sha256_file(Path(__file__).with_name('verify_media.py'))):
            raise ValueError('media verification missing or stale')
        decoded={r['clip']:r for r in checked['videos']}
        if len(decoded)!=len(checked['videos']) or set(decoded)!={x['clip']['clip_id'] for x in clips}:
            raise ValueError('media verification clip inventory differs')
        for name in ('outcome.json','expert_trace.jsonl','expert_replay.npz'):
            if [v for k,v in review['source_sha256'].items() if k.endswith('/'+name)]!=[quality['artifact_sha256'][name]]:
                raise ValueError('video and measured episode differ')
        for item in clips:
            clip=item['clip'];proof=decoded[clip['clip_id']]
            if (proof['decoded_full'] is not True or proof['sha256']!=clip['video_sha256']
                    or proof['frame_count']!=clip['frame_count']):raise ValueError('video lacks full decoding')
            result['clips'][clip['clip_id']]={'video':relative(item['video_path']),'duration_s':clip['duration_s']}
        result['review']=relative(manifest)
        result['media_verification']=relative(receipt)
    if verify_hashes(episode,provenance['artifact_sha256']):raise ValueError('episode changed during report collection')
    if any(sha256_file(directory/name)!=digest for name,digest in report_hashes.items()):
        raise ValueError('numerical report changed during collection')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();config=json.loads(args.config.read_text());data=[collect(v) for v in config['variants']]
    escape=html.escape
    rows=[('七关节累计转角 · rad','travel',3),('方向反转 · 次（>1°）','reversals',0),
          ('模拟时间 · s','duration',2),('最高肘肩差 · mm','elbow_mm',2),
          ('总物理接触力峰值 · N','force_peak',3),('最短连续有效 HOLD · s','hold_min',2),
          ('实测原始加速度峰值 · rad/s²','acceleration_raw',3),
          ('实测原始 jerk 峰值 · rad/s³','jerk_raw',2),
          ('实测 0.1s 滤波 jerk 峰值 · rad/s³','jerk_filtered',2),
          ('自由运动命令 jerk 峰值 · rad/s³','free_jerk',3)]
    heads=''.join('<th>'+escape(d['label'])+'</th>' for d in data)
    table=''.join('<tr><th>'+label+'</th>'+''.join('<td>'+ (f'{d[key]:.{precision}f}' if d['complete'] else '部分记录，见报告')+'</td>' for d in data)+'</tr>' for label,key,precision in rows)
    table+='<tr><th>完成点数（控制器记录）</th>'+''.join(f'<td>{d["completed_points"]}/10</td>' for d in data)+'</tr>'
    decisions={'review_draft_preference_exceedances':'有草案超限，待选择标准',
        'reject_hard_failure':'硬检查失败，拒绝','pending_evaluation':'证据不足',
        'reject_selected_preference':'未满足已选偏好',
        'passes_selected_engineering_profile':'已选工程指标通过',
        'passes_legacy_gates_draft_preferences_within_bounds':'硬检查通过，草案范围内'}
    table+='<tr><th>数值判定</th>'+''.join('<td>'+escape(decisions.get(d['decision'],d['decision']))+'</td>' for d in data)+'</tr>'
    table+='<tr><th>自由命令检查</th>'+''.join('<td>'+('通过' if d['free_status']=='passed' else escape(d['free_status']+': '+'; '.join(d['free_failures'])))+'</td>' for d in data)+'</tr>'
    joints=''.join(f'<tr><th>J{j+1}</th>'+''.join('<td>'+ (f'{d["per_joint"][j]:.3f}' if d['complete'] else '—')+'</td>' for d in data)+'</tr>' for j in range(7))
    cards=''.join(f'<article><h2>{escape(d["label"])}</h2><p>{escape(d["description"])}</p><video id="v{i}" controls preload="metadata" playsinline></video><p id="t{i}"></p><a href="{escape(d["quality"])}">完整量化记录</a> · <a href="{escape(d["free"])}">自由运动检查</a></article>' for i,d in enumerate(data))
    notes=''.join('<li>'+escape(note)+'</li>' for note in config['notes'])
    failures=''.join('<li>'+escape(x)+'</li>' for x in config.get('attempt_notes',[]))
    payload=json.dumps(data,ensure_ascii=False,allow_nan=False).replace('</','<\\/')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>独立专家规划比较</title>
<style>body{margin:0;background:#f4f6f8;color:#203043;font:16px/1.65 system-ui,sans-serif}main{max-width:1440px;margin:auto;padding:30px}h1{margin:0;font-size:30px}h2{font-size:20px}article,.panel{background:white;border:1px solid #d8e0e8;border-radius:10px;padding:18px;margin:18px 0}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:18px}video{width:100%;background:#e5e9ed}button,select{font:inherit;padding:8px 12px;margin:5px;border:1px solid #8097aa;border-radius:5px}button{background:#17598b;color:white;cursor:pointer}th,td{text-align:left;padding:9px;border-bottom:1px solid #e0e6ec}table{width:100%;border-collapse:collapse}a{color:#17598b}.muted{color:#5c6d7b}.table{overflow:auto}</style>
<main><p class="muted">WAM · Isaac Sim · 仿真专家动作</p><h1>独立规划器与缓慢撤回</h1><p>保留患者、十点顺序和已确认的向下 7° 相机。两套新方法使用相同接触控制、撤回与限速。旧 B 保留作历史参照。</p>
<div class="panel"><label>点位 <select id="point"><option>R5</option><option>L1</option><option>R1</option></select></label><label>视角 <select id="view"><option value="posture">全臂</option><option value="robotcam">已确认相机</option></select></label><button id="play">从头同时原速播放</button><button id="pause">暂停</button><p class="muted">真实物理记录回放，1× 原速，不插值、不拉伸。片段长度不同，结束后保留末帧；精确停留与高频动态看 240 Hz 数值报告。</p></div><div class="cards">CARDS</div>
<h2>全十点量化</h2><div class="panel table"><table><thead><tr><th>指标</th>HEADS</tr></thead><tbody>TABLE</tbody></table><p class="muted">累计转角为七关节绝对转动总和。未完成十点的尝试不与全程总量比较。原始 jerk 与滤波值同时保留；工程草案不等于已选定的实机标准。</p></div>
<h2>各关节累计转角 · rad</h2><div class="panel table"><table><thead><tr><th>关节</th>HEADS</tr></thead><tbody>JOINTS</tbody></table></div><h2>结果与边界</h2><div class="panel"><ul>NOTES</ul><a href="RESULTS_ZH.md">完整结论、原始证据与复现</a></div>
<h2>R5 释放局部诊断</h2><figure class="panel"><img src="results/full10_independent_review/r5_release_diagnostics.png" alt="连续十点实验 R5 释放前后的原始采样诊断图" style="display:block;width:100%;height:auto" loading="lazy"><figcaption>连续十点 R5 释放前后 ±25 ms 原始采样，按实际力归零对齐；命令平滑不代表实测动态峰消失；局部图不能代表全程或确定因果。</figcaption><a href="results/full10_independent_review/r5_release_diagnostics.pdf">查看诊断图 PDF</a></figure>
<h2>保留的试验记录</h2><div class="panel"><ul>FAILURES</ul></div>
<script>const data=DATA;function load(){data.forEach((d,i)=>{let v=document.querySelector('#v'+i),c=d.clips[document.querySelector('#point').value+'_'+document.querySelector('#view').value];v.pause();v.hidden=!c;if(c){v.src=c.video;v.playbackRate=1;}document.querySelector('#t'+i).textContent=c?'片段 '+c.duration_s.toFixed(2)+' 秒 · 原速 1×':'本次尝试没有此点的有效审核片段';});}document.querySelector('#point').onchange=load;document.querySelector('#view').onchange=load;document.querySelector('#play').onclick=()=>document.querySelectorAll('video').forEach(v=>{if(!v.hidden){v.currentTime=0;v.playbackRate=1;v.play().catch(()=>{});}});document.querySelector('#pause').onclick=()=>document.querySelectorAll('video').forEach(v=>v.pause());load();</script></main></html>'''
    for key,value in {'CARDS':cards,'HEADS':heads,'TABLE':table,'JOINTS':joints,'NOTES':notes,'FAILURES':failures,'DATA':payload}.items():page=page.replace(key,value)
    (ROOT/'comparison.html').write_text(page)
    (ROOT/'results/comparison_summary.json').write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print('Built evidence-linked comparison.html')


if __name__=='__main__':main()
