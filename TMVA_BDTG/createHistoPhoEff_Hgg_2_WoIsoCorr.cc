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

  //  string fileName_EB_old = "/eos/user/k/kmondal/public/FLASHgg/PhotonIDMVA/PhaseISummer17/CMSSW_9_2_3_patch2/July24_2017/mvares_Hgg_phoId_80X_EB.root";
  //  string fileName_EB = "/afs/cern.ch/work/p/prrout/public/Myanalysis_2018/LowMassAnalysis_11_04_2018/PhotonIDMVA_Lowmass/mvares_Hgg_phoId_94X_EB_wophoisoCorr_RunIIFall17_3_0_0_MCv2_Lowmass.root";
  string fileName_EB = "/eos/user/p/prrout/public/Flashgg/PhotonIDMVA/stanalysis/PhotonIDMVA_StAnalysis_06102018/mvares_Hgg_phoId_94X_EB_wophoisoCorr_RunIIFall17_3_1_0_MCv2_SATrain_SATest_M95_wShowershape_nTree2k.root";
   string fileName_EE = "/eos/user/p/prrout/public/Flashgg/PhotonIDMVA/stanalysis/PhotonIDMVA_StAnalysis_06102018/mvares_Hgg_phoId_94X_EE_woCorrPhoIso_RunIIFall17_3_1_0_MCv2_SATrain_SATest_M95_wShowershape_nTree2k.root";  

  //  string fileName_EB = "/afs/cern.ch/work/p/prrout/public/Myanalysis_2018/LowMassAnalysis_11_04_2018/PhotonIDMVA_Lowmass/mvares_Hgg_phoId_94X_EB_wophoisoCorr_RunIIFall17_3_0_X_MCv2_LowmassTest_Mgg55_LowmassTrain_NewPUTarget_nTree2k.root";
  // string fileName_EE_old = "/eos/user/k/kmondal/public/FLASHgg/PhotonIDMVA/PhaseISummer17/CMSSW_9_2_3_patch2/July24_2017/mvares_Hgg_phoId_80X_EE.root";
  //string fileName_EE = "/afs/cern.ch/work/p/prrout/public/Myanalysis_2018/LowMassAnalysis_11_04_2018/PhotonIDMVA_Lowmass/mvares_Hgg_phoId_94X_EE_woCorrPhoIso_RunIIFall17_3_0_0_MCv2_Lowmass.root";
  //string fileName_EE = "/afs/cern.ch/work/p/prrout/public/Myanalysis_2018/LowMassAnalysis_11_04_2018/PhotonIDMVA_Lowmass/mvares_Hgg_phoId_94X_EE_woCorrPhoIso_RunIIFall17_3_0_X_MCv2_LowmassTest_LowmassTrain_M55_NewPUTarget_nTree2k.root";

  // Reading tree for EB
  TFile *file_EB = new TFile(fileName_EB.c_str());
  TTree *t_EB_s = (TTree*)file_EB->Get("promptPhotons");
  TTree *t_EB_b = (TTree*)file_EB->Get("fakePhotons");
 
 /*
  TFile *file_EB_old = new TFile(fileName_EB_old.c_str());
  TTree *t_EB_old_s = (TTree*)file_EB_old->Get("promptPhotons");
  TTree *t_EB_old_b = (TTree*)file_EB_old->Get("fakePhotons");
  */

  // EB, no mvares cut
  TH1F * h_pt_EB_s    = new TH1F ("h_pt_EB_s","h_pt_EB_s",25,0,250);
  TH1F * h_pt_EB_b    = new TH1F ("h_pt_EB_b","h_pt_EB_b",25,0,250);
  TH1F * h_nVtx_EB_s  = new TH1F ("h_nVtx_EB_s","h_nVtx_EB_s",40,0,40);
  TH1F * h_nVtx_EB_b  = new TH1F ("h_nVtx_EB_b","h_nVtx_EB_b",40,0,40);
  TH1F * h_eta_EB_s   = new TH1F ("h_eta_EB_s","h_eta_EB_s",50,-2.5,2.5);
  TH1F * h_eta_EB_b   = new TH1F ("h_eta_EB_b","h_eta_EB_b",50,-2.5,2.5);

  /*
  TH1F * h_pt_EB_old_s    = new TH1F ("h_pt_EB_old_s","h_pt_EB_old_s",25,0,250);
  TH1F * h_pt_EB_old_b    = new TH1F ("h_pt_EB_old_b","h_pt_EB_old_b",25,0,250);
  TH1F * h_nVtx_EB_old_s  = new TH1F ("h_nVtx_EB_old_s","h_nVtx_EB_old_s",40,0,40);
  TH1F * h_nVtx_EB_old_b  = new TH1F ("h_nVtx_EB_old_b","h_nVtx_EB_old_b",40,0,40);
  TH1F * h_eta_EB_old_s   = new TH1F ("h_eta_EB_old_s","h_eta_EB_old_s",50,-2.5,2.5);
  TH1F * h_eta_EB_old_b   = new TH1F ("h_eta_EB_old_b","h_eta_EB_old_b",50,-2.5,2.5);
  */

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
  
  /*
  h_pt_EB_old_s->Sumw2();
  h_pt_EB_old_b->Sumw2();
  h_nVtx_EB_old_s->Sumw2();
  h_nVtx_EB_old_b->Sumw2();
  h_eta_EB_old_s->Sumw2();
  h_eta_EB_old_b->Sumw2();
  */



  t_EB_s->Draw("pt>>h_pt_EB_s",EB_s,"goff");
  t_EB_b->Draw("pt>>h_pt_EB_b",EB_b,"goff");
  t_EB_s->Draw("nvtx>>h_nVtx_EB_s",EB_s,"goff");
  t_EB_b->Draw("nvtx>>h_nVtx_EB_b",EB_b,"goff");
  t_EB_s->Draw("scEta>>h_eta_EB_s",EB_s,"goff");
  t_EB_b->Draw("scEta>>h_eta_EB_b",EB_b,"goff");


  /*
  t_EB_old_s->Draw("pt>>h_pt_EB_old_s",EB_old_s,"goff");
  t_EB_old_b->Draw("pt>>h_pt_EB_old_b",EB_old_b,"goff");
  t_EB_old_s->Draw("nvtx>>h_nVtx_EB_old_s",EB_old_s,"goff");
  t_EB_old_b->Draw("nvtx>>h_nVtx_EB_old_b",EB_old_b,"goff");
  t_EB_old_s->Draw("scEta>>h_eta_EB_old_s",EB_old_s,"goff");
  t_EB_old_b->Draw("scEta>>h_eta_EB_old_b",EB_old_b,"goff");

  */
 
  //EB, mvares cut
  TH1F * h_pt_EB_Cut_s    = new TH1F ("h_pt_EB_Cut_s","h_pt_EB_Cut_s",25,0,250);
  TH1F * h_pt_EB_Cut_b    = new TH1F ("h_pt_EB_Cut_b","h_pt_EB_Cut_b",25,0,250);
  TH1F * h_nVtx_EB_Cut_s  = new TH1F ("h_nVtx_EB_Cut_s","h_nVtx_EB_Cut_s",40,0,40);
  TH1F * h_nVtx_EB_Cut_b  = new TH1F ("h_nVtx_EB_Cut_b","h_nVtx_EB_Cut_b",40,0,40);
  TH1F * h_eta_EB_Cut_s   = new TH1F ("h_eta_EB_Cut_s","h_eta_EB_Cut_s",50,-2.5,2.5);
  TH1F * h_eta_EB_Cut_b   = new TH1F ("h_eta_EB_Cut_b","h_eta_EB_Cut_b",50,-2.5,2.5);


  /*
  TH1F * h_pt_EB_old_Cut_s    = new TH1F ("h_pt_EB_old_Cut_s","h_pt_EB_old_Cut_s",25,0,250);
  TH1F * h_pt_EB_old_Cut_b    = new TH1F ("h_pt_EB_old_Cut_b","h_pt_EB_old_Cut_b",25,0,250);
  TH1F * h_nVtx_EB_old_Cut_s  = new TH1F ("h_nVtx_EB_old_Cut_s","h_nVtx_EB_old_Cut_s",40,0,40);
  TH1F * h_nVtx_EB_old_Cut_b  = new TH1F ("h_nVtx_EB_old_Cut_b","h_nVtx_EB_old_Cut_b",40,0,40);
  TH1F * h_eta_EB_old_Cut_s   = new TH1F ("h_eta_EB_old_Cut_s","h_eta_EB_old_Cut_s",50,-2.5,2.5);
  TH1F * h_eta_EB_old_Cut_b   = new TH1F ("h_eta_EB_old_Cut_b","h_eta_EB_old_Cut_b",50,-2.5,2.5);
  */


  //WP 90% Standard  Mass 08/10/2018
  TCut EB_Cut_s = "(mvares>0.21 && abs(scEta)<1.5 && pt > 18)";
  TCut EB_Cut_b = "(mvares>0.21 && abs(scEta)<1.5 && pt > 18)";
  
  /*
  TCut EB_old_Cut_s = "(mvares>0.1 && abs(scEta)<1.5 && pt > 18)";
  TCut EB_old_Cut_b = "(mvares>0.1 && abs(scEta)<1.5 && pt > 18)";
  */
  h_pt_EB_Cut_s->Sumw2();
  h_pt_EB_Cut_b->Sumw2();
  h_nVtx_EB_Cut_s->Sumw2();
  h_nVtx_EB_Cut_b->Sumw2();
  h_eta_EB_Cut_s->Sumw2();
  h_eta_EB_Cut_b->Sumw2();

  /*
  
  h_pt_EB_old_Cut_s->Sumw2();
  h_pt_EB_old_Cut_b->Sumw2();
  h_nVtx_EB_old_Cut_s->Sumw2();
  h_nVtx_EB_old_Cut_b->Sumw2();
  h_eta_EB_old_Cut_s->Sumw2();
  h_eta_EB_old_Cut_b->Sumw2();

  
  */
  t_EB_s->Draw("pt>>h_pt_EB_Cut_s",EB_Cut_s,"goff");
  t_EB_b->Draw("pt>>h_pt_EB_Cut_b",EB_Cut_b,"goff");
  t_EB_s->Draw("nvtx>>h_nVtx_EB_Cut_s",EB_Cut_s,"goff");
  t_EB_b->Draw("nvtx>>h_nVtx_EB_Cut_b",EB_Cut_b,"goff");
  t_EB_s->Draw("scEta>>h_eta_EB_Cut_s",EB_Cut_s,"goff");
  t_EB_b->Draw("scEta>>h_eta_EB_Cut_b",EB_Cut_b,"goff");
 
  /*
  t_EB_old_s->Draw("pt>>h_pt_EB_old_Cut_s",EB_old_Cut_s,"goff");
  t_EB_old_b->Draw("pt>>h_pt_EB_old_Cut_b",EB_old_Cut_b,"goff");
  t_EB_old_s->Draw("nvtx>>h_nVtx_EB_old_Cut_s",EB_old_Cut_s,"goff");
  t_EB_old_b->Draw("nvtx>>h_nVtx_EB_old_Cut_b",EB_old_Cut_b,"goff");
  t_EB_old_s->Draw("scEta>>h_eta_EB_old_Cut_s",EB_old_Cut_s,"goff");
  t_EB_old_b->Draw("scEta>>h_eta_EB_old_Cut_b",EB_old_Cut_b,"goff");
  
  */




  // Reading tree for EE
  TFile *file_EE = new TFile(fileName_EE.c_str());
  TTree *t_EE_s = (TTree*)file_EE->Get("promptPhotons");
  TTree *t_EE_b = (TTree*)file_EE->Get("fakePhotons");

  /*
  TFile *file_EE_old = new TFile(fileName_EE_old.c_str());
  TTree *t_EE_old_s = (TTree*)file_EE_old->Get("promptPhotons");
  TTree *t_EE_old_b = (TTree*)file_EE_old->Get("fakePhotons");

  */
  // EE, no mvares cut
  TH1F * h_pt_EE_s    = new TH1F ("h_pt_EE_s","h_pt_EE_s",25,0,250);
  TH1F * h_pt_EE_b    = new TH1F ("h_pt_EE_b","h_pt_EE_b",25,0,250);
  TH1F * h_nVtx_EE_s  = new TH1F ("h_nVtx_EE_s","h_nVtx_EE_s",40,0,40);
  TH1F * h_nVtx_EE_b  = new TH1F ("h_nVtx_EE_b","h_nVtx_EE_b",40,0,40);
  TH1F * h_eta_EE_s   = new TH1F ("h_eta_EE_s","h_eta_EE_s",50,-2.5,2.5);
  TH1F * h_eta_EE_b   = new TH1F ("h_eta_EE_b","h_eta_EE_b",50,-2.5,2.5);
  

  /*
  TH1F * h_pt_EE_old_s    = new TH1F ("h_pt_EE_old_s","h_pt_EE_old_s",25,0,250);
  TH1F * h_pt_EE_old_b    = new TH1F ("h_pt_EE_old_b","h_pt_EE_old_b",25,0,250);
  TH1F * h_nVtx_EE_old_s  = new TH1F ("h_nVtx_EE_old_s","h_nVtx_EE_old_s",40,0,40);
  TH1F * h_nVtx_EE_old_b  = new TH1F ("h_nVtx_EE_old_b","h_nVtx_EE_old_b",40,0,40);
  TH1F * h_eta_EE_old_s   = new TH1F ("h_eta_EE_old_s","h_eta_EE_old_s",50,-2.5,2.5);
  TH1F * h_eta_EE_old_b   = new TH1F ("h_eta_EE_old_b","h_eta_EE_old_b",50,-2.5,2.5);
  */

  h_pt_EE_s->Sumw2();
  h_pt_EE_b->Sumw2();
  h_nVtx_EE_s->Sumw2();
  h_nVtx_EE_b->Sumw2();
  h_eta_EE_s->Sumw2();
  h_eta_EE_b->Sumw2();

  /*
  h_pt_EE_old_s->Sumw2();
  h_pt_EE_old_b->Sumw2();
  h_nVtx_EE_old_s->Sumw2();
  h_nVtx_EE_old_b->Sumw2();
  h_eta_EE_old_s->Sumw2();
  h_eta_EE_old_b->Sumw2();
  */
 
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

  /*
  t_EE_old_s->Draw("pt>>h_pt_EE_old_s",EE_old_s,"goff");
  t_EE_old_b->Draw("pt>>h_pt_EE_old_b",EE_old_b,"goff");
  t_EE_old_s->Draw("nvtx>>h_nVtx_EE_old_s",EE_old_s,"goff");
  t_EE_old_b->Draw("nvtx>>h_nVtx_EE_old_b",EE_old_b,"goff");
  t_EE_old_s->Draw("scEta>>h_eta_EE_old_s",EE_old_s,"goff");
  t_EE_old_b->Draw("scEta>>h_eta_EE_old_b",EE_old_b,"goff");
  */

  //EE, mvares cut
  TH1F * h_pt_EE_Cut_s    = new TH1F ("h_pt_EE_Cut_s","h_pt_EE_Cut_s",25,0,250);
  TH1F * h_pt_EE_Cut_b    = new TH1F ("h_pt_EE_Cut_b","h_pt_EE_Cut_b",25,0,250);
  TH1F * h_nVtx_EE_Cut_s  = new TH1F ("h_nVtx_EE_Cut_s","h_nVtx_EE_Cut_s",40,0,40);
  TH1F * h_nVtx_EE_Cut_b  = new TH1F ("h_nVtx_EE_Cut_b","h_nVtx_EE_Cut_b",40,0,40);
  TH1F * h_eta_EE_Cut_s   = new TH1F ("h_eta_EE_Cut_s","h_eta_EE_Cut_s",50,-2.5,2.5);
  TH1F * h_eta_EE_Cut_b   = new TH1F ("h_eta_EE_Cut_b","h_eta_EE_Cut_b",50,-2.5,2.5);
  
  /*
  TH1F * h_pt_EE_old_Cut_s    = new TH1F ("h_pt_EE_old_Cut_s","h_pt_EE_old_Cut_s",25,0,250);
  TH1F * h_pt_EE_old_Cut_b    = new TH1F ("h_pt_EE_old_Cut_b","h_pt_EE_old_Cut_b",25,0,250);
  TH1F * h_nVtx_EE_old_Cut_s  = new TH1F ("h_nVtx_EE_old_Cut_s","h_nVtx_EE_old_Cut_s",40,0,40);
  TH1F * h_nVtx_EE_old_Cut_b  = new TH1F ("h_nVtx_EE_old_Cut_b","h_nVtx_EE_old_Cut_b",40,0,40);
  TH1F * h_eta_EE_old_Cut_s   = new TH1F ("h_eta_EE_old_Cut_s","h_eta_EE_old_Cut_s",50,-2.5,2.5);
  TH1F * h_eta_EE_old_Cut_b   = new TH1F ("h_eta_EE_old_Cut_b","h_eta_EE_old_Cut_b",50,-2.5,2.5);
  */

  //WP90% std mass 08/10/2018
  TCut EE_Cut_s = "(mvares>0.11 && abs(scEta)>1.5 && pt > 18)";
  TCut EE_Cut_b = "(mvares>0.11 && abs(scEta)>1.5 && pt > 18)";
  
  /*  
  TCut EE_old_Cut_s = "(mvares>0.15 && abs(scEta)>1.5 && pt > 18)";
  TCut EE_old_Cut_b = "(mvares>0.15 && abs(scEta)>1.5 && pt > 18)";
  */

  h_pt_EE_Cut_s->Sumw2();
  h_pt_EE_Cut_b->Sumw2();
  h_nVtx_EE_Cut_s->Sumw2();
  h_nVtx_EE_Cut_b->Sumw2();
  h_eta_EE_Cut_s->Sumw2();
  h_eta_EE_Cut_b->Sumw2();
 
  /*
  h_pt_EE_old_Cut_s->Sumw2();
  h_pt_EE_old_Cut_b->Sumw2();
  h_nVtx_EE_old_Cut_s->Sumw2();
  h_nVtx_EE_old_Cut_b->Sumw2();
  h_eta_EE_old_Cut_s->Sumw2();
  h_eta_EE_old_Cut_b->Sumw2();
  */

  t_EE_s->Draw("pt>>h_pt_EE_Cut_s",EE_Cut_s,"goff");
  t_EE_b->Draw("pt>>h_pt_EE_Cut_b",EE_Cut_b,"goff");
  t_EE_s->Draw("nvtx>>h_nVtx_EE_Cut_s",EE_Cut_s,"goff");
  t_EE_b->Draw("nvtx>>h_nVtx_EE_Cut_b",EE_Cut_b,"goff");
  t_EE_s->Draw("scEta>>h_eta_EE_Cut_s",EE_Cut_s,"goff");
  t_EE_b->Draw("scEta>>h_eta_EE_Cut_b",EE_Cut_b,"goff");

  /*
  t_EE_old_s->Draw("pt>>h_pt_EE_old_Cut_s",EE_old_Cut_s,"goff");
  t_EE_old_b->Draw("pt>>h_pt_EE_old_Cut_b",EE_old_Cut_b,"goff");
  t_EE_old_s->Draw("nvtx>>h_nVtx_EE_old_Cut_s",EE_old_Cut_s,"goff");
  t_EE_old_b->Draw("nvtx>>h_nVtx_EE_old_Cut_b",EE_old_Cut_b,"goff");
  t_EE_old_s->Draw("scEta>>h_eta_EE_old_Cut_s",EE_old_Cut_s,"goff");
  t_EE_old_b->Draw("scEta>>h_eta_EE_old_Cut_b",EE_old_Cut_b,"goff");
  */

  // effs EB
  TH1F * h_pt_EB_eff_s = new TH1F ("h_pt_EB_eff_s","h_pt_EB_eff_s",25,0,250);
  TH1F * h_pt_EB_eff_b = new TH1F ("h_pt_EB_eff_b","h_pt_EB_eff_b",25,0,250);
  TH1F * h_nVtx_EB_eff_s = new TH1F ("h_nVtx_EB_eff_s","h_nVtx_EB_eff_s",40,0,40);
  TH1F * h_nVtx_EB_eff_b = new TH1F ("h_nVtx_EB_eff_b","h_nVtx_EB_eff_b",40,0,40);
  TH1F * h_eta_EB_eff_s = new TH1F ("h_eta_EB_eff_s","h_eta_EB_eff_s",50,-2.5,2.5);
  TH1F * h_eta_EB_eff_b = new TH1F ("h_eta_EB_eff_b","h_eta_EB_eff_b",50,-2.5,2.5);
  
  /*
  TH1F * h_pt_EB_old_eff_s = new TH1F ("h_pt_EB_old_eff_s","h_pt_EB_old_eff_s",25,0,250);
  TH1F * h_pt_EB_old_eff_b = new TH1F ("h_pt_EB_old_eff_b","h_pt_EB_old_eff_b",25,0,250);
  TH1F * h_nVtx_EB_old_eff_s = new TH1F ("h_nVtx_EB_old_eff_s","h_nVtx_EB_old_eff_s",40,0,40);
  TH1F * h_nVtx_EB_old_eff_b = new TH1F ("h_nVtx_EB_old_eff_b","h_nVtx_EB_old_eff_b",40,0,40);
  TH1F * h_eta_EB_old_eff_s = new TH1F ("h_eta_EB_old_eff_s","h_eta_EB_old_eff_s",50,-2.5,2.5);
  TH1F * h_eta_EB_old_eff_b = new TH1F ("h_eta_EB_old_eff_b","h_eta_EB_old_eff_b",50,-2.5,2.5);
  */

  h_pt_EB_eff_s->Sumw2();
  h_pt_EB_eff_b->Sumw2();
  h_nVtx_EB_eff_s->Sumw2();
  h_nVtx_EB_eff_b->Sumw2();
  h_eta_EB_eff_s->Sumw2();
  h_eta_EB_eff_b->Sumw2();

  /*
  h_pt_EB_old_eff_s->Sumw2();
  h_pt_EB_old_eff_b->Sumw2();
  h_nVtx_EB_old_eff_s->Sumw2();
  h_nVtx_EB_old_eff_b->Sumw2();
  h_eta_EB_old_eff_s->Sumw2();
  h_eta_EB_old_eff_b->Sumw2();
  */
   
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

  /*
  h_pt_EB_old_eff_s = (TH1F*)h_pt_EB_old_Cut_s->Clone("h_pt_EB_old_eff_s");
  h_pt_EB_old_eff_s->Divide(h_pt_EB_old_s);
  h_pt_EB_old_eff_b = (TH1F*)h_pt_EB_old_Cut_b->Clone("h_pt_EB_old_eff_b");
  h_pt_EB_old_eff_b->Divide(h_pt_EB_old_b);
  h_nVtx_EB_old_eff_s = (TH1F*)h_nVtx_EB_old_Cut_s->Clone("h_nVtx_EB_old_eff_s");
  h_nVtx_EB_old_eff_s->Divide(h_nVtx_EB_old_s);
  h_nVtx_EB_old_eff_b = (TH1F*)h_nVtx_EB_old_Cut_b->Clone("h_nVtx_old_EB_eff_b");
  h_nVtx_EB_old_eff_b->Divide(h_nVtx_EB_old_b);
  h_eta_EB_old_eff_s = (TH1F*)h_eta_EB_old_Cut_s->Clone("h_eta_EB_old_eff_s");
  h_eta_EB_old_eff_s->Divide(h_eta_EB_old_s);
  h_eta_EB_old_eff_b = (TH1F*)h_eta_EB_old_Cut_b->Clone("h_eta_old_EB_eff_b");
  h_eta_EB_old_eff_b->Divide(h_eta_EB_old_b);
  */

  // effs EE
  TH1F * h_pt_EE_eff_s = new TH1F ("h_pt_EE_eff_s","h_pt_EE_eff_s",25,0,250);
  TH1F * h_pt_EE_eff_b = new TH1F ("h_pt_EE_eff_b","h_pt_EE_eff_b",25,0,250);
  TH1F * h_nVtx_EE_eff_s = new TH1F ("h_nVtx_EE_eff_s","h_nVtx_EE_eff_s",40,0,40);
  TH1F * h_nVtx_EE_eff_b = new TH1F ("h_nVtx_EE_eff_b","h_nVtx_EE_eff_b",40,0,40);
  TH1F * h_eta_EE_eff_s = new TH1F ("h_eta_EE_eff_s","h_eta_EE_eff_s",50,-2.5,2.5);
  TH1F * h_eta_EE_eff_b = new TH1F ("h_eta_EE_eff_b","h_eta_EE_eff_b",50,-2.5,2.5);

  /*
  TH1F * h_pt_EE_old_eff_s = new TH1F ("h_pt_EE_old_eff_s","h_pt_EE_old_eff_s",25,0,250);
  TH1F * h_pt_EE_old_eff_b = new TH1F ("h_pt_EE_old_eff_b","h_pt_EE_old_eff_b",25,0,250);
  TH1F * h_nVtx_EE_old_eff_s = new TH1F ("h_nVtx_EE_old_eff_s","h_nVtx_EE_old_eff_s",40,0,40);
  TH1F * h_nVtx_EE_old_eff_b = new TH1F ("h_nVtx_EE_old_eff_b","h_nVtx_EE_old_eff_b",40,0,40);
  TH1F * h_eta_EE_old_eff_s = new TH1F ("h_eta_EE_old_eff_s","h_eta_EE_old_eff_s",50,-2.5,2.5);
  TH1F * h_eta_EE_old_eff_b = new TH1F ("h_eta_EE_old_eff_b","h_eta_EE_old_eff_b",50,-2.5,2.5);
  */

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

  /*
  h_pt_EE_old_eff_s->Sumw2();
  h_pt_EE_old_eff_b->Sumw2();
  h_nVtx_EE_old_eff_s->Sumw2();
  h_nVtx_EE_old_eff_b->Sumw2();
  h_eta_EE_old_eff_s->Sumw2();
  h_eta_EE_old_eff_b->Sumw2();
  */

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

  /*
  h_pt_EE_old_eff_s = (TH1F*)h_pt_EE_old_Cut_s->Clone("h_pt_EE_old_eff_s");
  h_pt_EE_old_eff_s->Divide(h_pt_EE_old_s);
  h_pt_EE_old_eff_b = (TH1F*)h_pt_EE_old_Cut_b->Clone("h_pt_EE_old_eff_b");
  h_pt_EE_old_eff_b->Divide(h_pt_EE_old_b);
  h_nVtx_EE_old_eff_s = (TH1F*)h_nVtx_EE_old_Cut_s->Clone("h_nVtx_EE_old_eff_s");
  h_nVtx_EE_old_eff_s->Divide(h_nVtx_EE_old_s);
  h_nVtx_EE_old_eff_b = (TH1F*)h_nVtx_EE_old_Cut_b->Clone("h_nVtx_EE_old_eff_b");
  h_nVtx_EE_old_eff_b->Divide(h_nVtx_EE_old_b);
  h_eta_EE_old_eff_s = (TH1F*)h_eta_EE_old_Cut_s->Clone("h_eta_EE_old_eff_s");
  h_eta_EE_old_eff_s->Divide(h_eta_EE_old_s);
  h_eta_EE_old_eff_b = (TH1F*)h_eta_EE_old_Cut_b->Clone("h_eta_EE_old_eff_b");
  h_eta_EE_old_eff_b->Divide(h_eta_EE_old_b);
  */

  ratio_pt_eff_EE_76_80_s = (TH1F*)h_pt_EE_eff_s->Clone("ratio_pt_eff_EE_76_80_s");
  ratio_pt_eff_EE_76_80_s->Divide(h_pt_EE_eff_b);
  // ratio_pt_eff_EE_76_80_b= (TH1F*)h_pt_EE_old_eff_s->Clone("ratio_pt_eff_EE_76_80_b");
  // ratio_pt_eff_EE_76_80_b->Divide(h_pt_EE_old_eff_b);

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

  TCanvas * can_pt_eff_EB = new TCanvas("can_pt_eff_EB","can_pt_eff_EB",800,600);
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

  /*
  h_pt_EB_old_eff_s->SetLineColor(kAzure-4);
  h_pt_EB_old_eff_s->SetMarkerStyle(20);
  h_pt_EB_old_eff_s->SetMarkerSize(0.7);
  h_pt_EB_old_eff_s->SetMarkerColor(kAzure-4);
  h_pt_EB_old_eff_s->Draw("Psame");
  */

  h_pt_EB_eff_b->SetLineColor(kRed);
  h_pt_EB_eff_b->SetMarkerStyle(20);
  h_pt_EB_eff_b->SetMarkerSize(0.7);
  h_pt_EB_eff_b->SetMarkerColor(kRed);
  h_pt_EB_eff_b->Draw("Psame");

  /*
  h_pt_EB_old_eff_b->SetLineColor(kRed-9);
  h_pt_EB_old_eff_b->SetMarkerStyle(20);
  h_pt_EB_old_eff_b->SetMarkerSize(0.7);
  h_pt_EB_old_eff_b->SetMarkerColor(kRed-9);
  h_pt_EB_old_eff_b->Draw("Psame");
  */

  leg_EB->AddEntry(h_pt_EB_eff_s, "Signal: 94X","P");
  leg_EB->AddEntry(h_pt_EB_eff_b, "Bkg : 94X","P");
  // leg->AddEntry(h_pt_EB_old_eff_s, "Signal : 80X","P");
  // leg->AddEntry(h_pt_EB_old_eff_b, "Bkg : 80X","P");
  leg_EB->Draw("same");

  can_pt_eff_EB->Update();
  can_pt_eff_EB->Modified();

  can_pt_eff_EB->SaveAs("phoId_eff_vs_pt_barrel_Hgg.pdf");
  can_pt_eff_EB->SaveAs("phoId_eff_vs_pt_barrel_Hgg.png");
  can_pt_eff_EB->SaveAs("phoId_eff_vs_pt_barrel_Hgg.root");

  // eff vs nVtx, EB
  TCanvas * can_nVtx_eff_EB = new TCanvas("can_nVtx_eff_EB","can_nVtx_eff_EB",800,600);
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

  /*
  h_nVtx_EB_old_eff_s->SetLineColor(kAzure-4);
  h_nVtx_EB_old_eff_s->SetMarkerStyle(20);
  h_nVtx_EB_old_eff_s->SetMarkerSize(0.7);
  h_nVtx_EB_old_eff_s->SetMarkerColor(kAzure-4);
  h_nVtx_EB_old_eff_s->Draw("Psame");
  */

  h_nVtx_EB_eff_b->SetLineColor(kRed);
  h_nVtx_EB_eff_b->SetMarkerStyle(20);
  h_nVtx_EB_eff_b->SetMarkerSize(0.7);
  h_nVtx_EB_eff_b->SetMarkerColor(kRed);
  h_nVtx_EB_eff_b->Draw("Psame");
   
  /*
  h_nVtx_EB_old_eff_b->SetLineColor(kRed-9);
  h_nVtx_EB_old_eff_b->SetMarkerStyle(20);
  h_nVtx_EB_old_eff_b->SetMarkerSize(0.7);
  h_nVtx_EB_old_eff_b->SetMarkerColor(kRed-9);
  h_nVtx_EB_old_eff_b->Draw("Psame");
  */
  leg_EB->Draw("same");

  can_nVtx_eff_EB->Update();
  can_nVtx_eff_EB->Modified();

  can_nVtx_eff_EB->SaveAs("phoId_eff_vs_nVtx_barrel_Hgg.pdf");
  can_nVtx_eff_EB->SaveAs("phoId_eff_vs_nVtx_barrel_Hgg.png");
  can_nVtx_eff_EB->SaveAs("phoId_eff_vs_nVtx_barrel_Hgg.root");

  // eff vs eta, EB
   TCanvas * can_eta_eff_EB = new TCanvas("can_eta_eff_EB","can_eta_eff_EB",800,600);
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
  
  /*
  h_eta_EB_old_eff_s->SetLineColor(kAzure-4);
  h_eta_EB_old_eff_s->SetMarkerStyle(20);
  h_eta_EB_old_eff_s->SetMarkerSize(0.7);
  h_eta_EB_old_eff_s->SetMarkerColor(kAzure-4);
  h_eta_EB_old_eff_s->Draw("Psame");
  */

  h_eta_EB_eff_b->SetLineColor(kRed);
  h_eta_EB_eff_b->SetMarkerStyle(20);
  h_eta_EB_eff_b->SetMarkerSize(0.7);
  h_eta_EB_eff_b->SetMarkerColor(kRed);
  h_eta_EB_eff_b->Draw("Psame");
 
  /*
  h_eta_EB_old_eff_b->SetLineColor(kRed-9);
  h_eta_EB_old_eff_b->SetMarkerStyle(20);
  h_eta_EB_old_eff_b->SetMarkerSize(0.7);
  h_eta_EB_old_eff_b->SetMarkerColor(kRed-9);
  h_eta_EB_old_eff_b->Draw("Psame");
  */
  leg_EB->Draw("same");

  can_eta_eff_EB->Update();
  can_eta_eff_EB->Modified();

  can_eta_eff_EB->SaveAs("phoId_eff_vs_eta_barrel_Hgg.pdf");
  can_eta_eff_EB->SaveAs("phoId_eff_vs_eta_barrel_Hgg.png");
  can_eta_eff_EB->SaveAs("phoId_eff_vs_eta_barrel_Hgg.root"); 

  // eff vs eta, EE                                                                                                                            
  TCanvas * can_eta_eff_EE = new TCanvas("can_eta_eff_EE","can_eta_eff_EE",800,600);
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

  /*                                                                                                                                             h_eta_EE_old_eff_s->SetLineColor(kAzure-4);                                                                                                    h_eta_EE_old_eff_s->SetMarkerStyle(20);                                                                                                       h_eta_EE_old_eff_s->SetMarkerSize(0.7);                                                                                                       h_eta_EE_old_eff_s->SetMarkerColor(kAzure-4);                                                                                                 h_eta_EE_old_eff_s->Draw("Psame");
  */

  h_eta_EE_eff_b->SetLineColor(kRed);
  h_eta_EE_eff_b->SetMarkerStyle(20);
  h_eta_EE_eff_b->SetMarkerSize(0.7);
  h_eta_EE_eff_b->SetMarkerColor(kRed);
  h_eta_EE_eff_b->Draw("Psame");

  /*                                                                                                                                           
  h_eta_EE_old_eff_b->SetLineColor(kRed-9);                                                                                                    
  h_eta_EE_old_eff_b->SetMarkerStyle(20);                                                                                                      
  h_eta_EE_old_eff_b->SetMarkerSize(0.7);                                                                                                      
  h_eta_EE_old_eff_b->SetMarkerColor(kRed-9);                                                                                                  
  h_eta_EE_old_eff_b->Draw("Psame");                                                                                                           
  */
  leg_EE->AddEntry(h_eta_EE_eff_s, "Signal: 94X","P");
  leg_EE->AddEntry(h_eta_EE_eff_b, "Bkg : 94X","P");


  leg_EE->Draw("same");

  can_eta_eff_EE->Update();
  can_eta_eff_EE->Modified();

  can_eta_eff_EE->SaveAs("phoId_eff_vs_eta_endcap_Hgg.pdf");
  can_eta_eff_EE->SaveAs("phoId_eff_vs_eta_endcap_Hgg.png");
  can_eta_eff_EE->SaveAs("phoId_eff_vs_eta_endcap_Hgg.root");



  // eff vs pT, EE 
  TCanvas * can_pt_eff_EE = new TCanvas("can_pt_eff_EE","can_pt_eff_EE",800,600);
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

  /*
  h_pt_EE_old_eff_s->SetLineColor(kAzure-4);
  h_pt_EE_old_eff_s->SetMarkerStyle(20);
  h_pt_EE_old_eff_s->SetMarkerSize(0.7);
  h_pt_EE_old_eff_s->SetMarkerColor(kAzure-4);
  h_pt_EE_old_eff_s->Draw("Psame");
  */
  h_pt_EE_eff_b->SetLineColor(kRed);
  h_pt_EE_eff_b->SetMarkerStyle(20);
  h_pt_EE_eff_b->SetMarkerSize(0.7);
  h_pt_EE_eff_b->SetMarkerColor(kRed);
  h_pt_EE_eff_b->Draw("Psame");
 
  /*
  h_pt_EE_old_eff_b->SetLineColor(kRed-9);
  h_pt_EE_old_eff_b->SetMarkerStyle(20);
  h_pt_EE_old_eff_b->SetMarkerSize(0.7);
  h_pt_EE_old_eff_b->SetMarkerColor(kRed-9);
  h_pt_EE_old_eff_b->Draw("Psame");
  */
  //leg_EE->AddEntry(h_pt_EE_eff_s, "Signal: 94X","P");
  //leg_EE->AddEntry(h_pt_EE_eff_b, "Bkg : 94X","P");
  
  /*
  leg_EE->AddEntry(h_pt_EE_old_eff_s, "Signal : 80X","P");
  leg_EE->AddEntry(h_pt_EE_old_eff_b, "Bkg : 80X","P");
  */
  leg_EE->Draw("same");

  can_pt_eff_EE->Update();
  can_pt_eff_EE->Modified();

  can_pt_eff_EE->SaveAs("phoId_eff_vs_pt_endcap_Hgg.pdf");
  can_pt_eff_EE->SaveAs("phoId_eff_vs_pt_endcap_Hgg.png");
  can_pt_eff_EE->SaveAs("phoId_eff_vs_pt_endcap_Hgg.root");

  // eff vs nVtx, EE
  TCanvas * can_nVtx_eff_EE = new TCanvas("can_nVtx_eff_EE","can_nVtx_eff_EE",800,600);
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

  /*
  h_nVtx_EE_old_eff_s->SetLineColor(kAzure-4);
  h_nVtx_EE_old_eff_s->SetMarkerStyle(20);
  h_nVtx_EE_old_eff_s->SetMarkerSize(0.7);
  h_nVtx_EE_old_eff_s->SetMarkerColor(kAzure-4);
  h_nVtx_EE_old_eff_s->Draw("Psame");
  */

  h_nVtx_EE_eff_b->SetLineColor(kRed);
  h_nVtx_EE_eff_b->SetMarkerStyle(20);
  h_nVtx_EE_eff_b->SetMarkerSize(0.7);
  h_nVtx_EE_eff_b->SetMarkerColor(kRed);
  h_nVtx_EE_eff_b->Draw("Psame");

  /*
  h_nVtx_EE_old_eff_b->SetLineColor(kRed-9);
  h_nVtx_EE_old_eff_b->SetMarkerStyle(20);
  h_nVtx_EE_old_eff_b->SetMarkerSize(0.7);
  h_nVtx_EE_old_eff_b->SetMarkerColor(kRed-9);
  h_nVtx_EE_old_eff_b->Draw("Psame");
  */

  leg_EE->Draw("same");

  can_nVtx_eff_EE->Update();
  can_nVtx_eff_EE->Modified();

  can_nVtx_eff_EE->SaveAs("phoId_eff_vs_nVtx_endcap_Hgg.pdf");
  can_nVtx_eff_EE->SaveAs("phoId_eff_vs_nVtx_endcap_Hgg.png");
  can_nVtx_eff_EE->SaveAs("phoId_eff_vs_nVtx_endcap_Hgg.root");

  // eff vs eta
  TCanvas * can_eta_eff = new TCanvas("can_eta_eff","can_eta_eff",800,600);
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
  
  /*
  h_eta_EB_old_eff_s->SetLineColor(kAzure-4);
  h_eta_EB_old_eff_s->SetMarkerStyle(20);
  h_eta_EB_old_eff_s->SetMarkerSize(0.7);
  h_eta_EB_old_eff_s->SetMarkerColor(kAzure-4);
  h_eta_EB_old_eff_s->Draw("Psame");
  */

  h_eta_EB_eff_b->SetLineColor(kRed);
  h_eta_EB_eff_b->SetMarkerStyle(20);
  h_eta_EB_eff_b->SetMarkerSize(0.7);
  h_eta_EB_eff_b->SetMarkerColor(kRed);
  h_eta_EB_eff_b->Draw("Psame");

  /*
  h_eta_EB_old_eff_b->SetLineColor(kRed-9);
  h_eta_EB_old_eff_b->SetMarkerStyle(20);
  h_eta_EB_old_eff_b->SetMarkerSize(0.7);
  h_eta_EB_old_eff_b->SetMarkerColor(kRed-9);
  h_eta_EB_old_eff_b->Draw("Psame");
  */

  h_eta_EE_eff_s->SetLineColor(kBlue);
  h_eta_EE_eff_s->SetMarkerStyle(20);
  h_eta_EE_eff_s->SetMarkerSize(0.7);
  h_eta_EE_eff_s->SetMarkerColor(kBlue);
  h_eta_EE_eff_s->Draw("Psame");

  /*
  h_eta_EE_old_eff_s->SetLineColor(kAzure-4);
  h_eta_EE_old_eff_s->SetMarkerStyle(20);
  h_eta_EE_old_eff_s->SetMarkerSize(0.7);
  h_eta_EE_old_eff_s->SetMarkerColor(kAzure-4);
  h_eta_EE_old_eff_s->Draw("Psame");
  */
  h_eta_EE_eff_b->SetLineColor(kRed);
  h_eta_EE_eff_b->SetMarkerStyle(20);
  h_eta_EE_eff_b->SetMarkerSize(0.7);
  h_eta_EE_eff_b->SetMarkerColor(kRed);
  h_eta_EE_eff_b->Draw("Psame");
 
  /*
  h_eta_EE_old_eff_b->SetLineColor(kRed-9);
  h_eta_EE_old_eff_b->SetMarkerStyle(20);
  h_eta_EE_old_eff_b->SetMarkerSize(0.7);
  h_eta_EE_old_eff_b->SetMarkerColor(kRed-9);
  h_eta_EE_old_eff_b->Draw("Psame");
  */

  TLegend *leg = new TLegend(0.4,0.4,0.6,0.6);
  leg->SetBorderSize(0);
  leg->SetFillColor(0);
  leg->SetHeader("");
  leg->AddEntry(h_eta_EE_eff_s, "Signal: 94X","P");
  leg->AddEntry(h_eta_EE_eff_b, "Bkg : 94X","P");

  leg->Draw("same");

  can_eta_eff->Update();
  can_eta_eff->Modified();

  can_eta_eff->SaveAs("phoId_eff_vs_eta_Hgg.pdf");
  can_eta_eff->SaveAs("phoId_eff_vs_eta_Hgg.png");
  can_eta_eff->SaveAs("phoId_eff_vs_eta_Hgg.root");

}
