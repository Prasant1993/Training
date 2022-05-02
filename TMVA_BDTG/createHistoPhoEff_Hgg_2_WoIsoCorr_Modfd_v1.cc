#include "TROOT.h"
#include "TKey.h"
#include "TFile.h"
#include "TSystem.h"
#include "TCanvas.h"
#include "TTree.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TObjArray.h"
#include "THStack.h"
#include "TLegend.h"
#include "TEfficiency.h"
#include "TGraphAsymmErrors.h"
#include "TF1.h"
#include "TMath.h"
#include "TCut.h"
#include "TPaletteAxis.h"

#include "TMVA/Reader.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <string>
#include <algorithm>

     
void saveHistoToFile(){

   

  // TMVA barrel
  string fileName_St_EB = "Ntuple_UL2017/mvares_Hgg_phoId_EB_LR_0p1_UL2017_SATrain_UL2017_SATest_BDTG_14052020.root";  


  // Xgboost barrel
  string fileName_EB = "Ntuple_UL2017/mvares_Hgg_phoId_EB_UL2017_XGB_SATrain_UL2017_SATest_BDT_14052020.root";  

  // TMVA endcap
  string fileName_St_EE = "Ntuple_UL2017/mvares_Hgg_phoId_EE_LR_0p1_UL2017_SATrain_UL2017_SATest_BDTG_14052020.root";  

  // Xgboost endcap
  string fileName_EE = "Ntuple_UL2017/mvares_Hgg_phoId_EE_UL2017_XGB_SATrain_UL2017_SATest_BDT_14052020.root"; 

  // Reading tree for EB
  TFile *file_EB = new TFile(fileName_EB.c_str());
  TTree *t_EB_s = (TTree*)file_EB->Get("promptPhotons");
  TTree *t_EB_b = (TTree*)file_EB->Get("fakePhotons");
 
 
  TFile *file_EB_St = new TFile(fileName_St_EB.c_str());
  TTree *t_EB_St_s = (TTree*)file_EB_St->Get("promptPhotons");
  TTree *t_EB_St_b = (TTree*)file_EB_St->Get("fakePhotons");
  

  // EB, no mvares cut
  TH1F * h_pt_EB_s    = new TH1F ("h_pt_EB_s","h_pt_EB_s",25,0,250);
  TH1F * h_pt_EB_b    = new TH1F ("h_pt_EB_b","h_pt_EB_b",25,0,250);
  TH1F * h_nVtx_EB_s  = new TH1F ("h_nVtx_EB_s","h_nVtx_EB_s",80,0,80);
  TH1F * h_nVtx_EB_b  = new TH1F ("h_nVtx_EB_b","h_nVtx_EB_b",80,0,80);
  TH1F * h_eta_EB_s   = new TH1F ("h_eta_EB_s","h_eta_EB_s",50,-2.5,2.5);
  TH1F * h_eta_EB_b   = new TH1F ("h_eta_EB_b","h_eta_EB_b",50,-2.5,2.5);

  
  TH1F * h_pt_EB_St_s    = new TH1F ("h_pt_EB_St_s","h_pt_EB_St_s",25,0,250);
  TH1F * h_pt_EB_St_b    = new TH1F ("h_pt_EB_St_b","h_pt_EB_St_b",25,0,250);
  TH1F * h_nVtx_EB_St_s  = new TH1F ("h_nVtx_EB_St_s","h_nVtx_EB_St_s",80,0,80);
  TH1F * h_nVtx_EB_St_b  = new TH1F ("h_nVtx_EB_St_b","h_nVtx_EB_St_b",80,0,80);
  TH1F * h_eta_EB_St_s   = new TH1F ("h_eta_EB_St_s","h_eta_EB_St_s",50,-2.5,2.5);
  TH1F * h_eta_EB_St_b   = new TH1F ("h_eta_EB_St_b","h_eta_EB_St_b",50,-2.5,2.5);
  

   TCut EB_s = "(mvares>-1 && abs(scEta)<1.5 && pt > 18)";
   TCut EB_b = "(mvares>-1 && abs(scEta)<1.5 && pt > 18)";

  /*

  TCut EB_old_s = "(mvares>-2 && abs(scEta)<1.5 && pt > 18)";
  TCut EB_old_b = "(mvares>-2 && abs(scEta)<1.5 && pt > 18)";
  */


  h_pt_EB_s->Sumw2();
  h_pt_EB_b->Sumw2();
  h_nVtx_EB_s->Sumw2();
  h_nVtx_EB_b->Sumw2();
  h_eta_EB_s->Sumw2();
  h_eta_EB_b->Sumw2();
  
  
  h_pt_EB_St_s->Sumw2();
  h_pt_EB_St_b->Sumw2();
  h_nVtx_EB_St_s->Sumw2();
  h_nVtx_EB_St_b->Sumw2();
  h_eta_EB_St_s->Sumw2();
  h_eta_EB_St_b->Sumw2();
  



  t_EB_s->Draw("pt>>h_pt_EB_s",EB_s,"goff");
  t_EB_b->Draw("pt>>h_pt_EB_b",EB_b,"goff");
  t_EB_s->Draw("nvtx>>h_nVtx_EB_s",EB_s,"goff");
  t_EB_b->Draw("nvtx>>h_nVtx_EB_b",EB_b,"goff");
  t_EB_s->Draw("scEta>>h_eta_EB_s",EB_s,"goff");
  t_EB_b->Draw("scEta>>h_eta_EB_b",EB_b,"goff");


  
  t_EB_St_s->Draw("pt>>h_pt_EB_St_s",EB_s,"goff");
  t_EB_St_b->Draw("pt>>h_pt_EB_St_b",EB_b,"goff");
  t_EB_St_s->Draw("nvtx>>h_nVtx_EB_St_s",EB_s,"goff");
  t_EB_St_b->Draw("nvtx>>h_nVtx_EB_St_b",EB_b,"goff");
  t_EB_St_s->Draw("scEta>>h_eta_EB_St_s",EB_s,"goff");
  t_EB_St_b->Draw("scEta>>h_eta_EB_St_b",EB_b,"goff");

  
 
  //EB, mvares cut
  TH1F * h_pt_EB_Cut_s    = new TH1F ("h_pt_EB_Cut_s","h_pt_EB_Cut_s",25,0,250);
  TH1F * h_pt_EB_Cut_b    = new TH1F ("h_pt_EB_Cut_b","h_pt_EB_Cut_b",25,0,250);
  TH1F * h_nVtx_EB_Cut_s  = new TH1F ("h_nVtx_EB_Cut_s","h_nVtx_EB_Cut_s",80,0,80);
  TH1F * h_nVtx_EB_Cut_b  = new TH1F ("h_nVtx_EB_Cut_b","h_nVtx_EB_Cut_b",80,0,80);
  TH1F * h_eta_EB_Cut_s   = new TH1F ("h_eta_EB_Cut_s","h_eta_EB_Cut_s",50,-2.5,2.5);
  TH1F * h_eta_EB_Cut_b   = new TH1F ("h_eta_EB_Cut_b","h_eta_EB_Cut_b",50,-2.5,2.5);


  
  TH1F * h_pt_EB_St_Cut_s    = new TH1F ("h_pt_EB_St_Cut_s","h_pt_EB_St_Cut_s",25,0,250);
  TH1F * h_pt_EB_St_Cut_b    = new TH1F ("h_pt_EB_St_Cut_b","h_pt_EB_St_Cut_b",25,0,250);
  TH1F * h_nVtx_EB_St_Cut_s  = new TH1F ("h_nVtx_EB_St_Cut_s","h_nVtx_EB_St_Cut_s",80,0,80);
  TH1F * h_nVtx_EB_St_Cut_b  = new TH1F ("h_nVtx_EB_St_Cut_b","h_nVtx_EB_St_Cut_b",80,0,80);
  TH1F * h_eta_EB_St_Cut_s   = new TH1F ("h_eta_EB_St_Cut_s","h_eta_EB_St_Cut_s",50,-2.5,2.5);
  TH1F * h_eta_EB_St_Cut_b   = new TH1F ("h_eta_EB_St_Cut_b","h_eta_EB_St_Cut_b",50,-2.5,2.5);
  

   

  // WP 80% XGBoost : 0.0635868       0.8001  0.936413        0.77397  less than
   //WP 90% XGBoost : 0.119129        0.900057        0.880871        -0.00104641 less than 

  //for XGB wp 90  
  //sigEff[k] = 0.900002 bkgEff[k] = 0.0736679 with a cut at 0.550933
  TCut EB_Cut_s = "(mvares>0.55 && abs(scEta)<1.5 && pt > 18)";
  TCut EB_Cut_b = "(mvares>0.55 && abs(scEta)<1.5 && pt > 18)";

  // WP 80% TMVA  : 0.0771573       0.800019        0.922843        0.440932  less than  
   // WP 90% TMVA : 0.145476        0.900114        0.854524        0.00695358 less than

  //for TMVA wp 90
  //sigEff[k] = 0.900093 bkgEff[k] = 0.0829747 with a cut at 0.196956
  TCut EB_St_Cut_s = "(mvares>0.19 && abs(scEta)<1.5 && pt > 18)";
  TCut EB_St_Cut_b = "(mvares>0.19 && abs(scEta)<1.5 && pt > 18)";
  
  h_pt_EB_Cut_s->Sumw2();
  h_pt_EB_Cut_b->Sumw2();
  h_nVtx_EB_Cut_s->Sumw2();
  h_nVtx_EB_Cut_b->Sumw2();
  h_eta_EB_Cut_s->Sumw2();
  h_eta_EB_Cut_b->Sumw2();

  
  
  h_pt_EB_St_Cut_s->Sumw2();
  h_pt_EB_St_Cut_b->Sumw2();
  h_nVtx_EB_St_Cut_s->Sumw2();
  h_nVtx_EB_St_Cut_b->Sumw2();
  h_eta_EB_St_Cut_s->Sumw2();
  h_eta_EB_St_Cut_b->Sumw2();

  
  
  t_EB_s->Draw("pt>>h_pt_EB_Cut_s",EB_Cut_s,"goff");
  t_EB_b->Draw("pt>>h_pt_EB_Cut_b",EB_Cut_b,"goff");
  t_EB_s->Draw("nvtx>>h_nVtx_EB_Cut_s",EB_Cut_s,"goff");
  t_EB_b->Draw("nvtx>>h_nVtx_EB_Cut_b",EB_Cut_b,"goff");
  t_EB_s->Draw("scEta>>h_eta_EB_Cut_s",EB_Cut_s,"goff");
  t_EB_b->Draw("scEta>>h_eta_EB_Cut_b",EB_Cut_b,"goff");
 
  
  t_EB_St_s->Draw("pt>>h_pt_EB_St_Cut_s",EB_St_Cut_s,"goff");
  t_EB_St_b->Draw("pt>>h_pt_EB_St_Cut_b",EB_St_Cut_b,"goff");
  t_EB_St_s->Draw("nvtx>>h_nVtx_EB_St_Cut_s",EB_St_Cut_s,"goff");
  t_EB_St_b->Draw("nvtx>>h_nVtx_EB_St_Cut_b",EB_St_Cut_b,"goff");
  t_EB_St_s->Draw("scEta>>h_eta_EB_St_Cut_s",EB_St_Cut_s,"goff");
  t_EB_St_b->Draw("scEta>>h_eta_EB_St_Cut_b",EB_St_Cut_b,"goff");
  
  
  // Reading tree for EE
  TFile *file_EE = new TFile(fileName_EE.c_str());
  TTree *t_EE_s = (TTree*)file_EE->Get("promptPhotons");
  TTree *t_EE_b = (TTree*)file_EE->Get("fakePhotons");

  
  TFile *file_EE_St = new TFile(fileName_St_EE.c_str());
  TTree *t_EE_St_s = (TTree*)file_EE_St->Get("promptPhotons");
  TTree *t_EE_St_b = (TTree*)file_EE_St->Get("fakePhotons");

  
  // EE, no mvares cut
  TH1F * h_pt_EE_s    = new TH1F ("h_pt_EE_s","h_pt_EE_s",25,0,250);
  TH1F * h_pt_EE_b    = new TH1F ("h_pt_EE_b","h_pt_EE_b",25,0,250);
  TH1F * h_nVtx_EE_s  = new TH1F ("h_nVtx_EE_s","h_nVtx_EE_s",80,0,80);
  TH1F * h_nVtx_EE_b  = new TH1F ("h_nVtx_EE_b","h_nVtx_EE_b",80,0,80);
  TH1F * h_eta_EE_s   = new TH1F ("h_eta_EE_s","h_eta_EE_s",50,-2.5,2.5);
  TH1F * h_eta_EE_b   = new TH1F ("h_eta_EE_b","h_eta_EE_b",50,-2.5,2.5);
  

  
  TH1F * h_pt_EE_St_s    = new TH1F ("h_pt_EE_St_s","h_pt_EE_St_s",25,0,250);
  TH1F * h_pt_EE_St_b    = new TH1F ("h_pt_EE_St_b","h_pt_EE_St_b",25,0,250);
  TH1F * h_nVtx_EE_St_s  = new TH1F ("h_nVtx_EE_St_s","h_nVtx_EE_St_s",80,0,80);
  TH1F * h_nVtx_EE_St_b  = new TH1F ("h_nVtx_EE_St_b","h_nVtx_EE_St_b",80,0,80);
  TH1F * h_eta_EE_St_s   = new TH1F ("h_eta_EE_St_s","h_eta_EE_St_s",50,-2.5,2.5);
  TH1F * h_eta_EE_St_b   = new TH1F ("h_eta_EE_St_b","h_eta_EE_St_b",50,-2.5,2.5);
  

  h_pt_EE_s->Sumw2();
  h_pt_EE_b->Sumw2();
  h_nVtx_EE_s->Sumw2();
  h_nVtx_EE_b->Sumw2();
  h_eta_EE_s->Sumw2();
  h_eta_EE_b->Sumw2();

  
  h_pt_EE_St_s->Sumw2();
  h_pt_EE_St_b->Sumw2();
  h_nVtx_EE_St_s->Sumw2();
  h_nVtx_EE_St_b->Sumw2();
  h_eta_EE_St_s->Sumw2();
  h_eta_EE_St_b->Sumw2();
  
 
  TCut EE_s = "(mvares>-1 && abs(scEta)>1.5 && pt > 18)";
  TCut EE_b = "(mvares>-1 && abs(scEta)>1.5 && pt > 18)";

  /*
  TCut EE_old_s = "(mvares>-2 && abs(scEta)>1.5 && pt > 18)";
  TCut EE_old_b = "(mvares>-2 && abs(scEta)>1.5 && pt > 18)";
  */

  t_EE_s->Draw("pt>>h_pt_EE_s",EE_s,"goff");
  t_EE_b->Draw("pt>>h_pt_EE_b",EE_b,"goff");
  t_EE_s->Draw("nvtx>>h_nVtx_EE_s",EE_s,"goff");
  t_EE_b->Draw("nvtx>>h_nVtx_EE_b",EE_b,"goff");
  t_EE_s->Draw("scEta>>h_eta_EE_s",EE_s,"goff");
  t_EE_b->Draw("scEta>>h_eta_EE_b",EE_b,"goff");

  
  t_EE_St_s->Draw("pt>>h_pt_EE_St_s",EE_s,"goff");
  t_EE_St_b->Draw("pt>>h_pt_EE_St_b",EE_b,"goff");
  t_EE_St_s->Draw("nvtx>>h_nVtx_EE_St_s",EE_s,"goff");
  t_EE_St_b->Draw("nvtx>>h_nVtx_EE_St_b",EE_b,"goff");
  t_EE_St_s->Draw("scEta>>h_eta_EE_St_s",EE_s,"goff");
  t_EE_St_b->Draw("scEta>>h_eta_EE_St_b",EE_b,"goff");
  

  //EE, mvares cut
  TH1F * h_pt_EE_Cut_s    = new TH1F ("h_pt_EE_Cut_s","h_pt_EE_Cut_s",25,0,250);
  TH1F * h_pt_EE_Cut_b    = new TH1F ("h_pt_EE_Cut_b","h_pt_EE_Cut_b",25,0,250);
  TH1F * h_nVtx_EE_Cut_s  = new TH1F ("h_nVtx_EE_Cut_s","h_nVtx_EE_Cut_s",80,0,80);
  TH1F * h_nVtx_EE_Cut_b  = new TH1F ("h_nVtx_EE_Cut_b","h_nVtx_EE_Cut_b",80,0,80);
  TH1F * h_eta_EE_Cut_s   = new TH1F ("h_eta_EE_Cut_s","h_eta_EE_Cut_s",50,-2.5,2.5);
  TH1F * h_eta_EE_Cut_b   = new TH1F ("h_eta_EE_Cut_b","h_eta_EE_Cut_b",50,-2.5,2.5);
  
  
  TH1F * h_pt_EE_St_Cut_s    = new TH1F ("h_pt_EE_St_Cut_s","h_pt_EE_St_Cut_s",25,0,250);
  TH1F * h_pt_EE_St_Cut_b    = new TH1F ("h_pt_EE_St_Cut_b","h_pt_EE_St_Cut_b",25,0,250);
  TH1F * h_nVtx_EE_St_Cut_s  = new TH1F ("h_nVtx_EE_St_Cut_s","h_nVtx_EE_St_Cut_s",80,0,80);
  TH1F * h_nVtx_EE_St_Cut_b  = new TH1F ("h_nVtx_EE_St_Cut_b","h_nVtx_EE_St_Cut_b",80,0,80);
  TH1F * h_eta_EE_St_Cut_s   = new TH1F ("h_eta_EE_St_Cut_s","h_eta_EE_St_Cut_s",50,-2.5,2.5);
  TH1F * h_eta_EE_St_Cut_b   = new TH1F ("h_eta_EE_St_Cut_b","h_eta_EE_St_Cut_b",50,-2.5,2.5);
  
  // wp 80% XGB : 0.133279        0.800247        0.866721        0.77697
  // wp 90%  XGB :  0.231698        0.900024        0.768302        0.206957

  // for XGb wp 90
  //sigEff[k] = 0.900123 bkgEff[k] = 0.145738 with a cut at 0.743965
  TCut EE_Cut_s = "(mvares>0.74 && abs(scEta)>1.5 && pt > 18)";
  TCut EE_Cut_b = "(mvares>0.74 && abs(scEta)>1.5 && pt > 18)";
  
  // wp 80% TMVA : 0.162724        0.800124        0.837276        0.177956
  // wp 90% TMVA :0.277796        0.900121        0.722204        -0.214049
 

  // for TMVA wp 90
  // sigEff[k] = 0.900062 bkgEff[k] = 0.15668 with a cut at 0.00895358
  TCut EE_St_Cut_s = "(mvares>0.008 && abs(scEta)>1.5 && pt > 18)";
  TCut EE_St_Cut_b = "(mvares>0.008 && abs(scEta)>1.5 && pt > 18)";
  

  h_pt_EE_Cut_s->Sumw2();
  h_pt_EE_Cut_b->Sumw2();
  h_nVtx_EE_Cut_s->Sumw2();
  h_nVtx_EE_Cut_b->Sumw2();
  h_eta_EE_Cut_s->Sumw2();
  h_eta_EE_Cut_b->Sumw2();
 
  
  h_pt_EE_St_Cut_s->Sumw2();
  h_pt_EE_St_Cut_b->Sumw2();
  h_nVtx_EE_St_Cut_s->Sumw2();
  h_nVtx_EE_St_Cut_b->Sumw2();
  h_eta_EE_St_Cut_s->Sumw2();
  h_eta_EE_St_Cut_b->Sumw2();
  

  t_EE_s->Draw("pt>>h_pt_EE_Cut_s",EE_Cut_s,"goff");
  t_EE_b->Draw("pt>>h_pt_EE_Cut_b",EE_Cut_b,"goff");
  t_EE_s->Draw("nvtx>>h_nVtx_EE_Cut_s",EE_Cut_s,"goff");
  t_EE_b->Draw("nvtx>>h_nVtx_EE_Cut_b",EE_Cut_b,"goff");
  t_EE_s->Draw("scEta>>h_eta_EE_Cut_s",EE_Cut_s,"goff");
  t_EE_b->Draw("scEta>>h_eta_EE_Cut_b",EE_Cut_b,"goff");

  
  t_EE_St_s->Draw("pt>>h_pt_EE_St_Cut_s",EE_St_Cut_s,"goff");
  t_EE_St_b->Draw("pt>>h_pt_EE_St_Cut_b",EE_St_Cut_b,"goff");
  t_EE_St_s->Draw("nvtx>>h_nVtx_EE_St_Cut_s",EE_St_Cut_s,"goff");
  t_EE_St_b->Draw("nvtx>>h_nVtx_EE_St_Cut_b",EE_St_Cut_b,"goff");
  t_EE_St_s->Draw("scEta>>h_eta_EE_St_Cut_s",EE_St_Cut_s,"goff");
  t_EE_St_b->Draw("scEta>>h_eta_EE_St_Cut_b",EE_St_Cut_b,"goff");
  

  // effs EB
  TH1F * h_pt_EB_eff_s = new TH1F ("h_pt_EB_eff_s","h_pt_EB_eff_s",25,0,250);
  TH1F * h_pt_EB_eff_b = new TH1F ("h_pt_EB_eff_b","h_pt_EB_eff_b",25,0,250);
  TH1F * h_nVtx_EB_eff_s = new TH1F ("h_nVtx_EB_eff_s","h_nVtx_EB_eff_s",80,0,80);
  TH1F * h_nVtx_EB_eff_b = new TH1F ("h_nVtx_EB_eff_b","h_nVtx_EB_eff_b",80,0,80);
  TH1F * h_eta_EB_eff_s = new TH1F ("h_eta_EB_eff_s","h_eta_EB_eff_s",50,-2.5,2.5);
  TH1F * h_eta_EB_eff_b = new TH1F ("h_eta_EB_eff_b","h_eta_EB_eff_b",50,-2.5,2.5);
  
  
  TH1F * h_pt_EB_St_eff_s = new TH1F ("h_pt_EB_St_eff_s","h_pt_EB_St_eff_s",25,0,250);
  TH1F * h_pt_EB_St_eff_b = new TH1F ("h_pt_EB_St_eff_b","h_pt_EB_St_eff_b",25,0,250);
  TH1F * h_nVtx_EB_St_eff_s = new TH1F ("h_nVtx_EB_St_eff_s","h_nVtx_EB_St_eff_s",80,0,80);
  TH1F * h_nVtx_EB_St_eff_b = new TH1F ("h_nVtx_EB_St_eff_b","h_nVtx_EB_St_eff_b",80,0,80);
  TH1F * h_eta_EB_St_eff_s = new TH1F ("h_eta_EB_St_eff_s","h_eta_EB_St_eff_s",50,-2.5,2.5);
  TH1F * h_eta_EB_St_eff_b = new TH1F ("h_eta_EB_St_eff_b","h_eta_EB_St_eff_b",50,-2.5,2.5);
  

  h_pt_EB_eff_s->Sumw2();
  h_pt_EB_eff_b->Sumw2();
  h_nVtx_EB_eff_s->Sumw2();
  h_nVtx_EB_eff_b->Sumw2();
  h_eta_EB_eff_s->Sumw2();
  h_eta_EB_eff_b->Sumw2();

  
  h_pt_EB_St_eff_s->Sumw2();
  h_pt_EB_St_eff_b->Sumw2();
  h_nVtx_EB_St_eff_s->Sumw2();
  h_nVtx_EB_St_eff_b->Sumw2();
  h_eta_EB_St_eff_s->Sumw2();
  h_eta_EB_St_eff_b->Sumw2();
  
   
  h_pt_EB_eff_s = (TH1F*)h_pt_EB_Cut_s->Clone("h_pt_EB_eff_s");
  h_pt_EB_eff_s->Divide(h_pt_EB_s);
  h_pt_EB_eff_b = (TH1F*)h_pt_EB_Cut_b->Clone("h_pt_EB_eff_b");
  h_pt_EB_eff_b->Divide(h_pt_EB_b);
  h_nVtx_EB_eff_s = (TH1F*)h_nVtx_EB_Cut_s->Clone("h_nVtx_EB_eff_s");
  h_nVtx_EB_eff_s->Divide(h_nVtx_EB_s);
  h_nVtx_EB_eff_b = (TH1F*)h_nVtx_EB_Cut_b->Clone("h_nVtx_EB_eff_b");
  h_nVtx_EB_eff_b->Divide(h_nVtx_EB_b);
  h_eta_EB_eff_s = (TH1F*)h_eta_EB_Cut_s->Clone("h_eta_EB_eff_s");
  h_eta_EB_eff_s->Divide(h_eta_EB_s);
  h_eta_EB_eff_b = (TH1F*)h_eta_EB_Cut_b->Clone("h_eta_EB_eff_b");
  h_eta_EB_eff_b->Divide(h_eta_EB_b);

  
  h_pt_EB_St_eff_s = (TH1F*)h_pt_EB_St_Cut_s->Clone("h_pt_EB_St_eff_s");
  h_pt_EB_St_eff_s->Divide(h_pt_EB_St_s);
  h_pt_EB_St_eff_b = (TH1F*)h_pt_EB_St_Cut_b->Clone("h_pt_EB_St_eff_b");
  h_pt_EB_St_eff_b->Divide(h_pt_EB_St_b);
  h_nVtx_EB_St_eff_s = (TH1F*)h_nVtx_EB_St_Cut_s->Clone("h_nVtx_EB_St_eff_s");
  h_nVtx_EB_St_eff_s->Divide(h_nVtx_EB_St_s);
  h_nVtx_EB_St_eff_b = (TH1F*)h_nVtx_EB_St_Cut_b->Clone("h_nVtx_EB_St_eff_b");
  h_nVtx_EB_St_eff_b->Divide(h_nVtx_EB_St_b);
  h_eta_EB_St_eff_s = (TH1F*)h_eta_EB_St_Cut_s->Clone("h_eta_EB_St_eff_s");
  h_eta_EB_St_eff_s->Divide(h_eta_EB_St_s);
  h_eta_EB_St_eff_b = (TH1F*)h_eta_EB_St_Cut_b->Clone("h_eta_EB_St_eff_b");
  h_eta_EB_St_eff_b->Divide(h_eta_EB_St_b);
  

  // effs EE
  TH1F * h_pt_EE_eff_s = new TH1F ("h_pt_EE_eff_s","h_pt_EE_eff_s",25,0,250);
  TH1F * h_pt_EE_eff_b = new TH1F ("h_pt_EE_eff_b","h_pt_EE_eff_b",25,0,250);
  TH1F * h_nVtx_EE_eff_s = new TH1F ("h_nVtx_EE_eff_s","h_nVtx_EE_eff_s",80,0,80);
  TH1F * h_nVtx_EE_eff_b = new TH1F ("h_nVtx_EE_eff_b","h_nVtx_EE_eff_b",80,0,80);
  TH1F * h_eta_EE_eff_s = new TH1F ("h_eta_EE_eff_s","h_eta_EE_eff_s",50,-2.5,2.5);
  TH1F * h_eta_EE_eff_b = new TH1F ("h_eta_EE_eff_b","h_eta_EE_eff_b",50,-2.5,2.5);

  
  TH1F * h_pt_EE_St_eff_s = new TH1F ("h_pt_EE_St_eff_s","h_pt_EE_St_eff_s",25,0,250);
  TH1F * h_pt_EE_St_eff_b = new TH1F ("h_pt_EE_St_eff_b","h_pt_EE_St_eff_b",25,0,250);
  TH1F * h_nVtx_EE_St_eff_s = new TH1F ("h_nVtx_EE_St_eff_s","h_nVtx_EE_St_eff_s",80,0,80);
  TH1F * h_nVtx_EE_St_eff_b = new TH1F ("h_nVtx_EE_St_eff_b","h_nVtx_EE_St_eff_b",80,0,80);
  TH1F * h_eta_EE_St_eff_s = new TH1F ("h_eta_EE_St_eff_s","h_eta_EE_St_eff_s",50,-2.5,2.5);
  TH1F * h_eta_EE_St_eff_b = new TH1F ("h_eta_EE_St_eff_b","h_eta_EE_St_eff_b",50,-2.5,2.5);
  

  TH1F * ratio_pt_eff_EE_76_80_s = new TH1F("ratio_pt_eff_EE_76_80_s","ratio_pt_eff_EE_76_80_s",25,0,250);
  TH1F * ratio_pt_eff_EE_76_80_b = new TH1F("ratio_pt_eff_EE_76_80_b","ratio_pt_eff_EE_76_80_b",25,0,250);

  ratio_pt_eff_EE_76_80_s->Sumw2();
  ratio_pt_eff_EE_76_80_b->Sumw2();

  h_pt_EE_eff_s->Sumw2();
  h_pt_EE_eff_b->Sumw2();
  h_nVtx_EE_eff_s->Sumw2();
  h_nVtx_EE_eff_b->Sumw2();
  h_eta_EE_eff_s->Sumw2();
  h_eta_EE_eff_b->Sumw2();

  
  h_pt_EE_St_eff_s->Sumw2();
  h_pt_EE_St_eff_b->Sumw2();
  h_nVtx_EE_St_eff_s->Sumw2();
  h_nVtx_EE_St_eff_b->Sumw2();
  h_eta_EE_St_eff_s->Sumw2();
  h_eta_EE_St_eff_b->Sumw2();
  

  h_pt_EE_eff_s = (TH1F*)h_pt_EE_Cut_s->Clone("h_pt_EE_eff_s");
  h_pt_EE_eff_s->Divide(h_pt_EE_s);
  h_pt_EE_eff_b = (TH1F*)h_pt_EE_Cut_b->Clone("h_pt_EE_eff_b");
  h_pt_EE_eff_b->Divide(h_pt_EE_b);
  h_nVtx_EE_eff_s = (TH1F*)h_nVtx_EE_Cut_s->Clone("h_nVtx_EE_eff_s");
  h_nVtx_EE_eff_s->Divide(h_nVtx_EE_s);
  h_nVtx_EE_eff_b = (TH1F*)h_nVtx_EE_Cut_b->Clone("h_nVtx_EE_eff_b");
  h_nVtx_EE_eff_b->Divide(h_nVtx_EE_b);
  h_eta_EE_eff_s = (TH1F*)h_eta_EE_Cut_s->Clone("h_eta_EE_eff_s");
  h_eta_EE_eff_s->Divide(h_eta_EE_s);
  h_eta_EE_eff_b = (TH1F*)h_eta_EE_Cut_b->Clone("h_eta_EE_eff_b");
  h_eta_EE_eff_b->Divide(h_eta_EE_b);

  
  h_pt_EE_St_eff_s = (TH1F*)h_pt_EE_St_Cut_s->Clone("h_pt_EE_St_eff_s");
  h_pt_EE_St_eff_s->Divide(h_pt_EE_St_s);
  h_pt_EE_St_eff_b = (TH1F*)h_pt_EE_St_Cut_b->Clone("h_pt_EE_St_eff_b");
  h_pt_EE_St_eff_b->Divide(h_pt_EE_St_b);
  h_nVtx_EE_St_eff_s = (TH1F*)h_nVtx_EE_St_Cut_s->Clone("h_nVtx_EE_St_eff_s");
  h_nVtx_EE_St_eff_s->Divide(h_nVtx_EE_St_s);
  h_nVtx_EE_St_eff_b = (TH1F*)h_nVtx_EE_St_Cut_b->Clone("h_nVtx_EE_St_eff_b");
  h_nVtx_EE_St_eff_b->Divide(h_nVtx_EE_St_b);
  h_eta_EE_St_eff_s = (TH1F*)h_eta_EE_St_Cut_s->Clone("h_eta_EE_St_eff_s");
  h_eta_EE_St_eff_s->Divide(h_eta_EE_St_s);
  h_eta_EE_St_eff_b = (TH1F*)h_eta_EE_St_Cut_b->Clone("h_eta_EE_St_eff_b");
  h_eta_EE_St_eff_b->Divide(h_eta_EE_St_b);
 

  ratio_pt_eff_EE_76_80_s = (TH1F*)h_pt_EE_eff_s->Clone("ratio_pt_eff_EE_76_80_s");
  ratio_pt_eff_EE_76_80_s->Divide(h_pt_EE_eff_b);
  //ratio_pt_eff_EE_76_80_b= (TH1F*)h_pt_EE_St_eff_s->Clone("ratio_pt_eff_EE_76_80_b");
  //ratio_pt_eff_EE_76_80_b->Divide(h_pt_EE_St_eff_b);

  // plot eff histos:
  // eff vs pT, EB
  TLegend *leg_EB = new TLegend(0.4,0.4,0.6,0.6);
  leg_EB->SetBorderSize(0);
  leg_EB->SetFillColor(0);
  leg_EB->SetHeader(" EB :");

  TLegend *leg_EE = new TLegend(0.4,0.4,0.6,0.6);
  leg_EE->SetBorderSize(0);
  leg_EE->SetFillColor(0);
  leg_EE->SetHeader(" EE :");

  TCanvas * can_pt_eff_EB = new TCanvas("can_pt_eff_EB","can_pt_eff_EB",600,600);
  can_pt_eff_EB->SetGrid();

  h_pt_EB_eff_s->SetStats(0);
  h_pt_EB_eff_s->SetTitle("");
  h_pt_EB_eff_s->GetXaxis()->SetTitle("Pt (GeV)");
  h_pt_EB_eff_s->GetYaxis()->SetTitle("Efficiency");
  h_pt_EB_eff_s->GetYaxis()->SetRangeUser(0.,1.);
  h_pt_EB_eff_s->SetLineColor(kBlue);
  h_pt_EB_eff_s->SetMarkerStyle(20);
  h_pt_EB_eff_s->SetMarkerSize(0.7);
  h_pt_EB_eff_s->SetMarkerColor(kBlue);
  h_pt_EB_eff_s->Draw("P");

  
  h_pt_EB_St_eff_s->SetLineColor(kAzure-4);
  h_pt_EB_St_eff_s->SetMarkerStyle(20);
  h_pt_EB_St_eff_s->SetMarkerSize(0.7);
  h_pt_EB_St_eff_s->SetMarkerColor(kAzure-4);
  h_pt_EB_St_eff_s->Draw("Psame");
  

  h_pt_EB_eff_b->SetLineColor(kRed);
  h_pt_EB_eff_b->SetMarkerStyle(20);
  h_pt_EB_eff_b->SetMarkerSize(0.7);
  h_pt_EB_eff_b->SetMarkerColor(kRed);
  h_pt_EB_eff_b->Draw("Psame");

  
  h_pt_EB_St_eff_b->SetLineColor(kRed-9);
  h_pt_EB_St_eff_b->SetMarkerStyle(20);
  h_pt_EB_St_eff_b->SetMarkerSize(0.7);
  h_pt_EB_St_eff_b->SetMarkerColor(kRed-9);
  h_pt_EB_St_eff_b->Draw("Psame");
  

  leg_EB->AddEntry(h_pt_EB_eff_s, "Signal: UL17_XGBoost","P");
  leg_EB->AddEntry(h_pt_EB_eff_b, "Bkg : UL17_XGBoost","P");
  leg_EB->AddEntry(h_pt_EB_St_eff_s, "Signal : UL17_TMVA","P");
  leg_EB->AddEntry(h_pt_EB_St_eff_b, "Bkg : UL17_TMVA","P");
  leg_EB->Draw("same");

  TLatex *txt = new TLatex(0.2, 0.9, "");
  // txt->SetTextSize(0.05);                                                                                        
  txt->DrawLatexNDC(0.1, 0.91, "CMS #bf{#it{#scale[0.8]{Simulation Preliminary}}}");
  txt->DrawLatexNDC(0.76, 0.91, "#bf{13 TeV}");
  txt->Draw("same");


  can_pt_eff_EB->Update();
  can_pt_eff_EB->Modified();

  

  can_pt_eff_EB->SaveAs("phoId_eff_vs_pt_barrel_Hgg.pdf");
  can_pt_eff_EB->SaveAs("phoId_eff_vs_pt_barrel_Hgg.png");
  can_pt_eff_EB->SaveAs("phoId_eff_vs_pt_barrel_Hgg.root");

  // eff vs nVtx, EB
  TCanvas * can_nVtx_eff_EB = new TCanvas("can_nVtx_eff_EB","can_nVtx_eff_EB",600,600);
  can_nVtx_eff_EB->SetGrid();

  h_nVtx_EB_eff_s->SetStats(0);
  h_nVtx_EB_eff_s->SetTitle("");
  h_nVtx_EB_eff_s->GetXaxis()->SetTitle("number of vertices");
  h_nVtx_EB_eff_s->GetYaxis()->SetTitle("Efficiency");
  h_nVtx_EB_eff_s->GetYaxis()->SetRangeUser(0.,1.);
  h_nVtx_EB_eff_s->SetLineColor(kBlue);
  h_nVtx_EB_eff_s->SetMarkerStyle(20);
  h_nVtx_EB_eff_s->SetMarkerSize(0.7);
  h_nVtx_EB_eff_s->SetMarkerColor(kBlue);
  h_nVtx_EB_eff_s->Draw("P");

  
  h_nVtx_EB_St_eff_s->SetLineColor(kAzure-4);
  h_nVtx_EB_St_eff_s->SetMarkerStyle(20);
  h_nVtx_EB_St_eff_s->SetMarkerSize(0.7);
  h_nVtx_EB_St_eff_s->SetMarkerColor(kAzure-4);
  h_nVtx_EB_St_eff_s->Draw("Psame");
  

  h_nVtx_EB_eff_b->SetLineColor(kRed);
  h_nVtx_EB_eff_b->SetMarkerStyle(20);
  h_nVtx_EB_eff_b->SetMarkerSize(0.7);
  h_nVtx_EB_eff_b->SetMarkerColor(kRed);
  h_nVtx_EB_eff_b->Draw("Psame");
   
  
  h_nVtx_EB_St_eff_b->SetLineColor(kRed-9);
  h_nVtx_EB_St_eff_b->SetMarkerStyle(20);
  h_nVtx_EB_St_eff_b->SetMarkerSize(0.7);
  h_nVtx_EB_St_eff_b->SetMarkerColor(kRed-9);
  h_nVtx_EB_St_eff_b->Draw("Psame");
  
  leg_EB->Draw("same");

  txt->DrawLatexNDC(0.1, 0.91, "CMS #bf{#it{#scale[0.8]{Simulation Preliminary}}}");
  txt->DrawLatexNDC(0.76, 0.91, "#bf{13 TeV}");
  txt->Draw("same");

  can_nVtx_eff_EB->Update();
  can_nVtx_eff_EB->Modified();

  can_nVtx_eff_EB->SaveAs("phoId_eff_vs_nVtx_barrel_Hgg.pdf");
  can_nVtx_eff_EB->SaveAs("phoId_eff_vs_nVtx_barrel_Hgg.png");
  can_nVtx_eff_EB->SaveAs("phoId_eff_vs_nVtx_barrel_Hgg.root");

  // eff vs eta, EB
   TCanvas * can_eta_eff_EB = new TCanvas("can_eta_eff_EB","can_eta_eff_EB",600,600);
  can_eta_eff_EB->SetGrid();

  h_eta_EB_eff_s->SetStats(0);
  h_eta_EB_eff_s->SetTitle("");
  h_eta_EB_eff_s->GetXaxis()->SetTitle("supercluster eta");
  h_eta_EB_eff_s->GetYaxis()->SetTitle("Efficiency");
  h_eta_EB_eff_s->GetYaxis()->SetRangeUser(0.,1.);
  h_eta_EB_eff_s->SetLineColor(kBlue);
  h_eta_EB_eff_s->SetMarkerStyle(20);
  h_eta_EB_eff_s->SetMarkerSize(0.7);
  h_eta_EB_eff_s->SetMarkerColor(kBlue);
  h_eta_EB_eff_s->Draw("P");
  
  
  h_eta_EB_St_eff_s->SetLineColor(kAzure-4);
  h_eta_EB_St_eff_s->SetMarkerStyle(20);
  h_eta_EB_St_eff_s->SetMarkerSize(0.7);
  h_eta_EB_St_eff_s->SetMarkerColor(kAzure-4);
  h_eta_EB_St_eff_s->Draw("Psame");
  

  h_eta_EB_eff_b->SetLineColor(kRed);
  h_eta_EB_eff_b->SetMarkerStyle(20);
  h_eta_EB_eff_b->SetMarkerSize(0.7);
  h_eta_EB_eff_b->SetMarkerColor(kRed);
  h_eta_EB_eff_b->Draw("Psame");
 
  
  h_eta_EB_St_eff_b->SetLineColor(kRed-9);
  h_eta_EB_St_eff_b->SetMarkerStyle(20);
  h_eta_EB_St_eff_b->SetMarkerSize(0.7);
  h_eta_EB_St_eff_b->SetMarkerColor(kRed-9);
  h_eta_EB_St_eff_b->Draw("Psame");
  
  leg_EB->Draw("same");

  txt->DrawLatexNDC(0.1, 0.91, "CMS #bf{#it{#scale[0.8]{Simulation Preliminary}}}");
  txt->DrawLatexNDC(0.76, 0.91, "#bf{13 TeV}");
  txt->Draw("same");

  can_eta_eff_EB->Update();
  can_eta_eff_EB->Modified();

  can_eta_eff_EB->SaveAs("phoId_eff_vs_eta_barrel_Hgg.pdf");
  can_eta_eff_EB->SaveAs("phoId_eff_vs_eta_barrel_Hgg.png");
  can_eta_eff_EB->SaveAs("phoId_eff_vs_eta_barrel_Hgg.root"); 


  // eff vs eta, EE  
                                                                                                                          
  TCanvas * can_eta_eff_EE = new TCanvas("can_eta_eff_EE","can_eta_eff_EE",600,600);
  can_eta_eff_EE->SetGrid();

  h_eta_EE_eff_s->SetStats(0);
  h_eta_EE_eff_s->SetTitle("");
  h_eta_EE_eff_s->GetXaxis()->SetTitle("supercluster eta");
  h_eta_EE_eff_s->GetYaxis()->SetTitle("Efficiency");
  h_eta_EE_eff_s->GetYaxis()->SetRangeUser(0.,1.);
  h_eta_EE_eff_s->SetLineColor(kBlue);
  h_eta_EE_eff_s->SetMarkerStyle(20);
  h_eta_EE_eff_s->SetMarkerSize(0.7);
  h_eta_EE_eff_s->SetMarkerColor(kBlue);
  h_eta_EE_eff_s->Draw("P");

                                                                                                                                              
  h_eta_EE_St_eff_s->SetLineColor(kAzure-4);                                                                                                 
  h_eta_EE_St_eff_s->SetMarkerStyle(20);                                                                                                     
  h_eta_EE_St_eff_s->SetMarkerSize(0.7);                                                                                                     
  h_eta_EE_St_eff_s->SetMarkerColor(kAzure-4);                                                                                               
  h_eta_EE_St_eff_s->Draw("Psame");
  

  h_eta_EE_eff_b->SetLineColor(kRed);
  h_eta_EE_eff_b->SetMarkerStyle(20);
  h_eta_EE_eff_b->SetMarkerSize(0.7);
  h_eta_EE_eff_b->SetMarkerColor(kRed);
  h_eta_EE_eff_b->Draw("Psame");

                                                                                                                                   
 

  h_eta_EE_St_eff_b->SetLineColor(kRed-9);                                                                                                   
  h_eta_EE_St_eff_b->SetMarkerStyle(20);                                                                                                    
  h_eta_EE_St_eff_b->SetMarkerSize(0.7);
  h_eta_EE_St_eff_b->SetMarkerColor(kRed-9);
  h_eta_EE_St_eff_b->Draw("Psame");
                                                                                                           
  
  leg_EE->AddEntry(h_eta_EE_eff_s, "Signal: UL17_XGBoost","P");
  leg_EE->AddEntry(h_eta_EE_eff_b, "Bkg : UL17_XGBoost","P");
  leg_EE->AddEntry(h_eta_EE_St_eff_s, "Signal: UL17_TMVA","P");
  leg_EE->AddEntry(h_eta_EE_St_eff_b, "Bkg : UL17_TMVA","P");


  leg_EE->Draw("same");

  txt->DrawLatexNDC(0.1, 0.91, "CMS #bf{#it{#scale[0.8]{Simulation Preliminary}}}");
  txt->DrawLatexNDC(0.76, 0.91, "#bf{13 TeV}");
  txt->Draw("same");

  can_eta_eff_EE->Update();
  can_eta_eff_EE->Modified();

  can_eta_eff_EE->SaveAs("phoId_eff_vs_eta_endcap_Hgg.pdf");
  can_eta_eff_EE->SaveAs("phoId_eff_vs_eta_endcap_Hgg.png");
  can_eta_eff_EE->SaveAs("phoId_eff_vs_eta_endcap_Hgg.root");



  // eff vs pT, EE 
  TCanvas * can_pt_eff_EE = new TCanvas("can_pt_eff_EE","can_pt_eff_EE",600,600);
  can_pt_eff_EE->SetGrid();

  h_pt_EE_eff_s->SetStats(0);
  h_pt_EE_eff_s->SetTitle("");
  h_pt_EE_eff_s->GetXaxis()->SetTitle("Pt (GeV)");
  h_pt_EE_eff_s->GetYaxis()->SetTitle("Efficiency");
  h_pt_EE_eff_s->GetYaxis()->SetRangeUser(0.,1.);
  h_pt_EE_eff_s->SetLineColor(kBlue);
  h_pt_EE_eff_s->SetMarkerStyle(20);
  h_pt_EE_eff_s->SetMarkerSize(0.7);
  h_pt_EE_eff_s->SetMarkerColor(kBlue);
  h_pt_EE_eff_s->Draw("P");

  
  h_pt_EE_St_eff_s->SetLineColor(kAzure-4);
  h_pt_EE_St_eff_s->SetMarkerStyle(20);
  h_pt_EE_St_eff_s->SetMarkerSize(0.7);
  h_pt_EE_St_eff_s->SetMarkerColor(kAzure-4);
  h_pt_EE_St_eff_s->Draw("Psame");
  
  h_pt_EE_eff_b->SetLineColor(kRed);
  h_pt_EE_eff_b->SetMarkerStyle(20);
  h_pt_EE_eff_b->SetMarkerSize(0.7);
  h_pt_EE_eff_b->SetMarkerColor(kRed);
  h_pt_EE_eff_b->Draw("Psame");
 
  
  h_pt_EE_St_eff_b->SetLineColor(kRed-9);
  h_pt_EE_St_eff_b->SetMarkerStyle(20);
  h_pt_EE_St_eff_b->SetMarkerSize(0.7);
  h_pt_EE_St_eff_b->SetMarkerColor(kRed-9);
  h_pt_EE_St_eff_b->Draw("Psame");
  
  //leg_EE->AddEntry(h_pt_EE_eff_s, "Signal: 94X","P");
  //leg_EE->AddEntry(h_pt_EE_eff_b, "Bkg : 94X","P");
  
  /*
  leg_EE->AddEntry(h_pt_EE_old_eff_s, "Signal : 80X","P");
  leg_EE->AddEntry(h_pt_EE_old_eff_b, "Bkg : 80X","P");
  */
  leg_EE->Draw("same");

  txt->DrawLatexNDC(0.1, 0.91, "CMS #bf{#it{#scale[0.8]{Simulation Preliminary}}}");
  txt->DrawLatexNDC(0.76, 0.91, "#bf{13 TeV}");
  txt->Draw("same");

  can_pt_eff_EE->Update();
  can_pt_eff_EE->Modified();

  can_pt_eff_EE->SaveAs("phoId_eff_vs_pt_endcap_Hgg.pdf");
  can_pt_eff_EE->SaveAs("phoId_eff_vs_pt_endcap_Hgg.png");
  can_pt_eff_EE->SaveAs("phoId_eff_vs_pt_endcap_Hgg.root");

  // eff vs nVtx, EE
  TCanvas * can_nVtx_eff_EE = new TCanvas("can_nVtx_eff_EE","can_nVtx_eff_EE",600,600);
  can_nVtx_eff_EE->SetGrid();

  h_nVtx_EE_eff_s->SetStats(0);
  h_nVtx_EE_eff_s->SetTitle("");
  h_nVtx_EE_eff_s->GetXaxis()->SetTitle("number of vertices");
  h_nVtx_EE_eff_s->GetYaxis()->SetTitle("Efficiency");
  h_nVtx_EE_eff_s->GetYaxis()->SetRangeUser(0.,1.);
  h_nVtx_EE_eff_s->SetLineColor(kBlue);
  h_nVtx_EE_eff_s->SetMarkerStyle(20);
  h_nVtx_EE_eff_s->SetMarkerSize(0.7);
  h_nVtx_EE_eff_s->SetMarkerColor(kBlue);
  h_nVtx_EE_eff_s->Draw("P");

  
  h_nVtx_EE_St_eff_s->SetLineColor(kAzure-4);
  h_nVtx_EE_St_eff_s->SetMarkerStyle(20);
  h_nVtx_EE_St_eff_s->SetMarkerSize(0.7);
  h_nVtx_EE_St_eff_s->SetMarkerColor(kAzure-4);
  h_nVtx_EE_St_eff_s->Draw("Psame");
  

  h_nVtx_EE_eff_b->SetLineColor(kRed);
  h_nVtx_EE_eff_b->SetMarkerStyle(20);
  h_nVtx_EE_eff_b->SetMarkerSize(0.7);
  h_nVtx_EE_eff_b->SetMarkerColor(kRed);
  h_nVtx_EE_eff_b->Draw("Psame");

  
  h_nVtx_EE_St_eff_b->SetLineColor(kRed-9);
  h_nVtx_EE_St_eff_b->SetMarkerStyle(20);
  h_nVtx_EE_St_eff_b->SetMarkerSize(0.7);
  h_nVtx_EE_St_eff_b->SetMarkerColor(kRed-9);
  h_nVtx_EE_St_eff_b->Draw("Psame");
  

  leg_EE->Draw("same");
  txt->DrawLatexNDC(0.1, 0.91, "CMS #bf{#it{#scale[0.8]{Simulation Preliminary}}}");
  txt->DrawLatexNDC(0.76, 0.91, "#bf{13 TeV}");
  txt->Draw("same");


  can_nVtx_eff_EE->Update();
  can_nVtx_eff_EE->Modified();

  can_nVtx_eff_EE->SaveAs("phoId_eff_vs_nVtx_endcap_Hgg.pdf");
  can_nVtx_eff_EE->SaveAs("phoId_eff_vs_nVtx_endcap_Hgg.png");
  can_nVtx_eff_EE->SaveAs("phoId_eff_vs_nVtx_endcap_Hgg.root");

  // eff vs eta
  TCanvas * can_eta_eff = new TCanvas("can_eta_eff","can_eta_eff",600,600);
  can_eta_eff->SetGrid();

  h_eta_EB_eff_s->SetStats(0);
  h_eta_EB_eff_s->SetTitle("");
  h_eta_EB_eff_s->GetXaxis()->SetTitle("supercluster eta");
  h_eta_EB_eff_s->GetYaxis()->SetTitle("Efficiency");
  h_eta_EB_eff_s->GetYaxis()->SetRangeUser(0.,1.);
  h_eta_EB_eff_s->SetLineColor(kBlue);
  h_eta_EB_eff_s->SetMarkerStyle(20);
  h_eta_EB_eff_s->SetMarkerSize(0.7);
  h_eta_EB_eff_s->SetMarkerColor(kBlue);
  h_eta_EB_eff_s->Draw("P");
  
  
  h_eta_EB_St_eff_s->SetLineColor(kAzure-4);
  h_eta_EB_St_eff_s->SetMarkerStyle(20);
  h_eta_EB_St_eff_s->SetMarkerSize(0.7);
  h_eta_EB_St_eff_s->SetMarkerColor(kAzure-4);
  h_eta_EB_St_eff_s->Draw("Psame");
  

  h_eta_EB_eff_b->SetLineColor(kRed);
  h_eta_EB_eff_b->SetMarkerStyle(20);
  h_eta_EB_eff_b->SetMarkerSize(0.7);
  h_eta_EB_eff_b->SetMarkerColor(kRed);
  h_eta_EB_eff_b->Draw("Psame");

  
  h_eta_EB_St_eff_b->SetLineColor(kRed-9);
  h_eta_EB_St_eff_b->SetMarkerStyle(20);
  h_eta_EB_St_eff_b->SetMarkerSize(0.7);
  h_eta_EB_St_eff_b->SetMarkerColor(kRed-9);
  h_eta_EB_St_eff_b->Draw("Psame");
  

  h_eta_EE_eff_s->SetLineColor(kBlue);
  h_eta_EE_eff_s->SetMarkerStyle(20);
  h_eta_EE_eff_s->SetMarkerSize(0.7);
  h_eta_EE_eff_s->SetMarkerColor(kBlue);
  h_eta_EE_eff_s->Draw("Psame");

  
  h_eta_EE_St_eff_s->SetLineColor(kAzure-4);
  h_eta_EE_St_eff_s->SetMarkerStyle(20);
  h_eta_EE_St_eff_s->SetMarkerSize(0.7);
  h_eta_EE_St_eff_s->SetMarkerColor(kAzure-4);
  h_eta_EE_St_eff_s->Draw("Psame");
  
  h_eta_EE_eff_b->SetLineColor(kRed);
  h_eta_EE_eff_b->SetMarkerStyle(20);
  h_eta_EE_eff_b->SetMarkerSize(0.7);
  h_eta_EE_eff_b->SetMarkerColor(kRed);
  h_eta_EE_eff_b->Draw("Psame");
 
  
  h_eta_EE_St_eff_b->SetLineColor(kRed-9);
  h_eta_EE_St_eff_b->SetMarkerStyle(20);
  h_eta_EE_St_eff_b->SetMarkerSize(0.7);
  h_eta_EE_St_eff_b->SetMarkerColor(kRed-9);
  h_eta_EE_St_eff_b->Draw("Psame");
  

  TLegend *leg = new TLegend(0.4,0.4,0.6,0.6);
  leg->SetBorderSize(0);
  leg->SetFillColor(0);
  leg->SetHeader("");
  leg->AddEntry(h_eta_EE_eff_s, "Signal: UL17_XGBoost","P");
  leg->AddEntry(h_eta_EE_eff_b, "Bkg : UL17_XGBoost","P");
  leg->AddEntry(h_eta_EE_St_eff_s, "Signal: UL17_TMVA","P");
  leg->AddEntry(h_eta_EE_St_eff_b, "Bkg : UL17_TMVA","P");


  leg->Draw("same");

  txt->DrawLatexNDC(0.1, 0.91, "CMS #bf{#it{#scale[0.8]{Simulation Preliminary}}}");
  txt->DrawLatexNDC(0.76, 0.91, "#bf{13 TeV}");
  txt->Draw("same");

  can_eta_eff->Update();
  can_eta_eff->Modified();

  can_eta_eff->SaveAs("phoId_eff_vs_eta_Hgg.pdf");
  can_eta_eff->SaveAs("phoId_eff_vs_eta_Hgg.png");
  can_eta_eff->SaveAs("phoId_eff_vs_eta_Hgg.root");

}
