import {Trainer,TRAINING} from './training.ts';
import {benchmark} from './benchmark.ts';
import {validModel,type ControllerKind,type Model} from './policy.ts';
// Published direct-input controls (no connectome), first declared training seed.
const direct=async()=>{
 const models:Model[]=[];
 for(const kind of ['direct-8-12-3','direct-8-20-3'] as ControllerKind[])
  try{
   const m:unknown=await(await fetch(`/benchmarks/direct/${kind.slice(7)}/20260912/model.json`)).json();
   if(validModel(m,kind))models.push(m);
  }catch{/* Evaluate without that control if it cannot be loaded. */}
 return models;
};
self.onmessage=async(event:MessageEvent)=>{
 try{
  if(event.data.type==='benchmark'){
   if(!validModel(event.data.model))throw new Error('Invalid model');
   self.postMessage({type:'benchmark',result:benchmark(event.data.model,await direct())});return;
  }
  const trainer=new Trainer(Number(event.data.seed)||20260912);
  self.postMessage({type:'initial',model:trainer.champion});
  const run=()=>{
   try{
    const progress=trainer.step();self.postMessage({type:'progress',progress});
    if(trainer.generation<TRAINING.generations)setTimeout(run,30);
    else self.postMessage({type:'complete'});
   }catch(error){self.postMessage({type:'error',message:String(error)});}
  };setTimeout(run,30);
 }catch(error){self.postMessage({type:'error',message:String(error)});}
};
