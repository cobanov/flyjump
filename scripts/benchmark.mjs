import {readFileSync,writeFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {benchmark} from '../src/lib/benchmark.ts';
import {validModel} from '../src/lib/policy.ts';
const raw=readFileSync('public/benchmarks/model.json','utf8'),model=JSON.parse(raw);
if(!validModel(model))throw new Error('Invalid model');
// Direct-input controls from the same first declared training seed, not chosen by test score.
const direct=['direct-8-12-3','direct-8-20-3'].map(kind=>{
 const m=JSON.parse(readFileSync(`public/benchmarks/direct/${kind.slice(7)}/${model.trainingSeed}/model.json`,'utf8'));
 if(!validModel(m,kind))throw new Error(`Invalid ${kind} model`);
 return m;
});
const report={...benchmark(model,direct),modelSha256:createHash('sha256').update(raw).digest('hex')};
writeFileSync('public/benchmarks/benchmark.json',JSON.stringify(report,null,2));
console.table(report.results.map(({runs,...result})=>result));
