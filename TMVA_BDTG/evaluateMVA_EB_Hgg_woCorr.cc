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

void evaluateMVA_EB_Hgg_woCorr(){
  
  // UL16 SATest
  string treeFileName = "Ntuple/PreVFP_PostVFP_separate_training/Output_SinglePhoton_GJet_UL16_PreVFP_photonIsocorrectionEE_Added_Test_26042022.root";

  // extend space sample wHOE
  //string treeFileName = "/eos/user/p/prrout/public/Flashgg/PhotonIDMVA/Extended_Space_05062019/Ntuple_wHOE_08062019_corrected/out_singlephoton_ntuple_extended_Phasespace_Mgg55_08062019_Test.root";

  // Autumn18 LM test sample                                                                                                                 
    
  //string treeFileName = "Ntuple/output_ntuple_Autumn18GJet_Lowmass_18pT18_Test_13012020.root";
  // 2017 low mass test sample
  //string treeFileName = "/eos/user/p/prrout/public/Flashgg/PhotonIDMVA/Lowmass/PhotonIDMVA_Lowmass_03122018/Out_Singlephoton_Lowmass_photonIDMVA_wShowershape_LMTest_18pT18_RunIIFall17_3_1_0_03122018.root";

  // SA test sample  woHOE                                                                                                                  

  //string treeFileName = "/eos/user/p/prrout/public/Flashgg/PhotonIDMVA/stanalysis/PhotonIDMVA_StAnalysis_06102018/output_singlephoton_GJet_SATest_woShowershape_061018.root";

  // LM test sample wHOE                                                                                                                            //string treeFileName = "/eos/user/p/prrout/public/Flashgg/PhotonIDMVA/Lowmass/PhotonIDMVA_Lowmass_12042019/Ntuple/out_singlephoton_LMTest_ntuple_55Mgg120_3GjetSamples_woShowershape_12042019.root";                                                                                     
  // LM test sample wHOE
  //string treeFileName = "/eos/user/p/prrout/public/Flashgg/PhotonIDMVA/Lowmass/PhotonIDMVA_Lowmass_03122018/Out_Singlephoton_Lowmass_photonIDMVA_woShowershape_LMTest_18pT18_RunIIFall17_3_1_0_03122018.root";

  TFile *treeFile = new TFile(treeFileName.c_str());

  TTree *t_sig = (TTree*)treeFile->Get("promptPhotons");
  TTree *t_bkg = (TTree*)treeFile->Get("fakePhotons");

  Long64_t nEntries_sig = t_sig->GetEntries();
  Long64_t nEntries_bkg = t_bkg->GetEntries();

  string outputFileName = "Ntuple/PreVFP_PostVFP_separate_training/mvares_Hgg_phoId_EB_UL16_TMVA_SATrain_PreVFP_UL16_SATest_PreVFP.root";

  //output file create
  TFile *outputFile = new TFile (outputFileName.c_str(),"recreate");
  TTree *outputTree_s = new TTree("promptPhotons","promptPhotons");
  TTree *outputTree_b = new TTree("fakePhotons","fakePhotons");

  //output vars:

  //sig:
  float out_SCRawE_s = 999.;
  float out_full_r9_s = 999.;
  float out_sigmaIetaIeta_s = 999.;
  float out_etaWidth_s = 999.;
  float out_phiWidth_s = 999.;
  float out_covIEtaIPhi_s = 999.;
  float out_s4_s = 999.;
  float out_isoPhotons_s = -999.;
  float out_isoChargedHad_s = -999.;
  float out_chgIsoWrtWorstVtx_s = -999.;
  float out_scEta_s = -999.;
  float out_rho_s = -999.;
  float out_scPhi_s = -999.;

  float out_esEffSigmaRR_s = -999.;
  float out_mvares_s = 999.;
  float out_weight_s = 999.;

  float out_E1x3_s = 999.;
  float out_E5x5_s = 999.;
  float out_E2x5_s = 999.;
  float out_EsEnergy_ov_scRawEnergy_s = 999.;
  float out_Pt_s = 999.;
  float out_Phi_s = 999.;
  float out_Eta_s = 999.;

  //bkg:
  float out_SCRawE_b = 999.;
  float out_full_r9_b = 999.;
  float out_sigmaIetaIeta_b = 999.;
  float out_etaWidth_b = 999.;
  float out_phiWidth_b = 999.;
  float out_covIEtaIPhi_b = 999.;
  float out_s4_b = 999.;
  float out_isoPhotons_b = -999.;
  float out_isoChargedHad_b = -999.;
  float out_chgIsoWrtWorstVtx_b = -999.;
  float out_scEta_b = -999.;
  float out_rho_b = -999.;
  float out_scPhi_b = -999.;
  float out_esEffSigmaRR_b = -999.;
  float out_mvares_b = 999.;
  float out_weight_b = 999.;

  float out_E1x3_b = 999.;
  float out_E5x5_b = 999.;
  float out_E2x5_b = 999.;
  float out_EsEnergy_ov_scRawEnergy_b = 999.;
  float out_Pt_b = 999.;
  float out_Phi_b = 999.;
  float out_Eta_b = 999.;

  float out_PtvsEtaWeight_s = 999.;
  float out_PtvsEtaWeight_b = 999.;
  float out_RhoRew_s = 999.;
  float out_RhoRew_b = 999.;

  int out_nvtx_s = 999.;
  int out_nvtx_b = 999.;
  float out_npu_s = 999.;
  float out_npu_b = 999.;

  //output file branches
  outputTree_s->Branch("SCRawE",&out_SCRawE_s);
  outputTree_s->Branch("r9",&out_full_r9_s);
  outputTree_s->Branch("sigmaIetaIeta",&out_sigmaIetaIeta_s);
  outputTree_s->Branch("etaWidth",&out_etaWidth_s);
  outputTree_s->Branch("phiWidth",&out_phiWidth_s);
  outputTree_s->Branch("covIEtaIPhi",&out_covIEtaIPhi_s);
  outputTree_s->Branch("s4",&out_s4_s);
  outputTree_s->Branch("phoIso03",&out_isoPhotons_s);
  outputTree_s->Branch("chgIsoWrtChosenVtx",&out_isoChargedHad_s);
  outputTree_s->Branch("chgIsoWrtWorstVtx",&out_chgIsoWrtWorstVtx_s);
  outputTree_s->Branch("scEta",&out_scEta_s);
  //outputTree_s->Branch("scPhi",&out_scPhi_s);
  outputTree_s->Branch("rho",&out_rho_s);
  outputTree_s->Branch("mvares",&out_mvares_s);
  outputTree_s->Branch("weight",&out_weight_s);
  outputTree_s->Branch("pt", &out_Pt_s );
  outputTree_s->Branch("phi", &out_Phi_s );
  outputTree_s->Branch("eta", &out_Eta_s );
  //outputTree_s->Branch("PtvsEtaWeight", &out_PtvsEtaWeight_s );
  //outputTree_s->Branch("rhoRew", &out_RhoRew_s );
  outputTree_s->Branch("nvtx", &out_nvtx_s );
  outputTree_s->Branch("npu", &out_npu_s );


  outputTree_b->Branch("SCRawE",&out_SCRawE_b);
  outputTree_b->Branch("r9",&out_full_r9_b);
  outputTree_b->Branch("sigmaIetaIeta",&out_sigmaIetaIeta_b);
  outputTree_b->Branch("etaWidth",&out_etaWidth_b);
  outputTree_b->Branch("phiWidth",&out_phiWidth_b);
  outputTree_b->Branch("covIEtaIPhi",&out_covIEtaIPhi_b);
  outputTree_b->Branch("s4",&out_s4_b);
  outputTree_b->Branch("phoIso03",&out_isoPhotons_b);
  outputTree_b->Branch("chgIsoWrtChosenVtx",&out_isoChargedHad_b);
  outputTree_b->Branch("chgIsoWrtWorstVtx",&out_chgIsoWrtWorstVtx_b);
  outputTree_b->Branch("scEta",&out_scEta_b);
  //outputTree_b->Branch("scPhi",&out_scPhi_b);
  outputTree_b->Branch("rho",&out_rho_b);
  outputTree_b->Branch("mvares",&out_mvares_b);
  outputTree_b->Branch("weight",&out_weight_b);
  outputTree_b->Branch("pt", &out_Pt_b );
  outputTree_b->Branch("phi", &out_Phi_b );
  outputTree_b->Branch("eta", &out_Eta_b );

  //outputTree_b->Branch("PtvsEtaWeight", &out_PtvsEtaWeight_b );
  //outputTree_b->Branch("rhoRew", &out_RhoRew_b );
  outputTree_b->Branch("nvtx", &out_nvtx_b );
  outputTree_b->Branch("npu", &out_npu_b );

  TMVA::Reader *photonIdMva_sig_ = new TMVA::Reader("!Color");
  TMVA::Reader *photonIdMva_bkg_ = new TMVA::Reader("!Color");

  photonIdMva_sig_->AddVariable("SCRawE", &out_SCRawE_s );
  photonIdMva_sig_->AddVariable("r9",&out_full_r9_s);
  photonIdMva_sig_->AddVariable("sigmaIetaIeta",&out_sigmaIetaIeta_s);
  photonIdMva_sig_->AddVariable("etaWidth",&out_etaWidth_s);
  photonIdMva_sig_->AddVariable("phiWidth",&out_phiWidth_s);
  photonIdMva_sig_->AddVariable("covIEtaIPhi",&out_covIEtaIPhi_s);
  photonIdMva_sig_->AddVariable("s4",&out_s4_s);
  photonIdMva_sig_->AddVariable("phoIso03",&out_isoPhotons_s);
  photonIdMva_sig_->AddVariable("chgIsoWrtChosenVtx",&out_isoChargedHad_s);
  photonIdMva_sig_->AddVariable("chgIsoWrtWorstVtx",&out_chgIsoWrtWorstVtx_s);
  photonIdMva_sig_->AddVariable("scEta",&out_scEta_s);
  //photonIdMva_sig_->AddVariable("scPhi",&out_scPhi_s);
  photonIdMva_sig_->AddVariable("rho",&out_rho_s);

  t_sig->SetBranchAddress("SCRawE",&out_SCRawE_s);
  t_sig->SetBranchAddress("r9",&out_full_r9_s);
  t_sig->SetBranchAddress("sigmaIetaIeta",&out_sigmaIetaIeta_s);
  t_sig->SetBranchAddress("etaWidth",&out_etaWidth_s);
  t_sig->SetBranchAddress("phiWidth",&out_phiWidth_s);
  t_sig->SetBranchAddress("covIEtaIPhi",&out_covIEtaIPhi_s);
  t_sig->SetBranchAddress("s4",&out_s4_s);
  t_sig->SetBranchAddress("phoIso03",&out_isoPhotons_s);
  t_sig->SetBranchAddress("chgIsoWrtChosenVtx",&out_isoChargedHad_s);
  t_sig->SetBranchAddress("chgIsoWrtWorstVtx",&out_chgIsoWrtWorstVtx_s);
  t_sig->SetBranchAddress("scEta",&out_scEta_s);
  //t_sig->SetBranchAddress("scPhi",&out_scPhi_s);
  t_sig->SetBranchAddress("rho",&out_rho_s);
  t_sig->SetBranchAddress("weight",&out_weight_s);
  t_sig->SetBranchAddress("pt", &out_Pt_s );
  t_sig->SetBranchAddress("phi", &out_Phi_s );
  t_sig->SetBranchAddress("eta", &out_Eta_s );
  //t_sig->SetBranchAddress("PtvsEtaWeight", &out_PtvsEtaWeight_s );
  //t_sig->SetBranchAddress("rhoRew", &out_RhoRew_s );
  t_sig->SetBranchAddress("nvtx", &out_nvtx_s );
  t_sig->SetBranchAddress("npu", &out_npu_s );

  photonIdMva_bkg_->AddVariable("SCRawE", &out_SCRawE_b );
  photonIdMva_bkg_->AddVariable("r9",&out_full_r9_b);
  photonIdMva_bkg_->AddVariable("sigmaIetaIeta",&out_sigmaIetaIeta_b);
  photonIdMva_bkg_->AddVariable("etaWidth",&out_etaWidth_b);
  photonIdMva_bkg_->AddVariable("phiWidth",&out_phiWidth_b);
  photonIdMva_bkg_->AddVariable("covIEtaIPhi",&out_covIEtaIPhi_b);
  photonIdMva_bkg_->AddVariable("s4",&out_s4_b);
  photonIdMva_bkg_->AddVariable("phoIso03",&out_isoPhotons_b);
  photonIdMva_bkg_->AddVariable("chgIsoWrtChosenVtx",&out_isoChargedHad_b);
  photonIdMva_bkg_->AddVariable("chgIsoWrtWorstVtx",&out_chgIsoWrtWorstVtx_b);
  photonIdMva_bkg_->AddVariable("scEta",&out_scEta_b);
  //photonIdMva_bkg_->AddVariable("scPhi",&out_scPhi_b);
  photonIdMva_bkg_->AddVariable("rho",&out_rho_b);

  t_bkg->SetBranchAddress("SCRawE",&out_SCRawE_b);
  t_bkg->SetBranchAddress("r9",&out_full_r9_b);
  t_bkg->SetBranchAddress("sigmaIetaIeta",&out_sigmaIetaIeta_b);
  t_bkg->SetBranchAddress("etaWidth",&out_etaWidth_b);
  t_bkg->SetBranchAddress("phiWidth",&out_phiWidth_b);
  t_bkg->SetBranchAddress("covIEtaIPhi",&out_covIEtaIPhi_b);
  t_bkg->SetBranchAddress("s4",&out_s4_b);
  t_bkg->SetBranchAddress("phoIso03",&out_isoPhotons_b);
  t_bkg->SetBranchAddress("chgIsoWrtChosenVtx",&out_isoChargedHad_b);
  t_bkg->SetBranchAddress("chgIsoWrtWorstVtx",&out_chgIsoWrtWorstVtx_b);
  t_bkg->SetBranchAddress("scEta",&out_scEta_b);
  //t_bkg->SetBranchAddress("scPhi",&out_scPhi_b);
  t_bkg->SetBranchAddress("rho",&out_rho_b);
  t_bkg->SetBranchAddress("weight",&out_weight_b);
  t_bkg->SetBranchAddress("pt", &out_Pt_b );
  t_bkg->SetBranchAddress("phi", &out_Phi_b );
  t_bkg->SetBranchAddress("eta", &out_Eta_b );
  // t_bkg->SetBranchAddress("PtvsEtaWeight", &out_PtvsEtaWeight_b );
  // t_bkg->SetBranchAddress("rhoRew", &out_RhoRew_b );

  t_bkg->SetBranchAddress("nvtx", &out_nvtx_b );
  t_bkg->SetBranchAddress("npu", &out_npu_b );


  // UL16 SA PreVFP training 

  photonIdMva_sig_->BookMVA("BDTG","/afs/cern.ch/work/p/prrout/public/Myanalysis_2022/BDT_training_UL16_Gjet_Standard_analysis_25022021/woCorr_EB_SATrain_BDTG_UL16_Added_photoniso_PreVFP/weights/PhoID_barrel_UL16_SA_Analysis_PreVFP_BDTG.weights.xml");                                
                            
  photonIdMva_bkg_->BookMVA("BDTG","/afs/cern.ch/work/p/prrout/public/Myanalysis_2022/BDT_training_UL16_Gjet_Standard_analysis_25022021/woCorr_EB_SATrain_BDTG_UL16_Added_photoniso_PreVFP/weights/PhoID_barrel_UL16_SA_Analysis_PreVFP_BDTG.weights.xml");
   


  // UL16 Standard analysis TMVA train                                                                                                       
  
  //photonIdMva_sig_->BookMVA("BDTG","/afs/cern.ch/work/p/prrout/public/Myanalysis_2022/BDT_training_UL16_Gjet_Standard_analysis_25022021/woCorr_EB_SATrain_BDTG_UL16/weights/PhoID_barrel_UL16_SA_Analysis_BDTG.weights.xml");                                             
  //photonIdMva_bkg_->BookMVA("BDTG","/afs/cern.ch/work/p/prrout/public/Myanalysis_2022/BDT_training_UL16_Gjet_Standard_analysis_25022021/woCorr_EB_SATrain_BDTG_UL16/weights/PhoID_barrel_UL16_SA_Analysis_BDTG.weights.xml");   

    // Rerco 16 SA TMVA train: Moriond17   
    //photonIdMva_sig_->BookMVA("BDTG","/afs/cern.ch/work/p/prrout/public/Myanalysis_2022/BDT_training_UL16_Gjet_Standard_analysis_25022021/Rereco_weight_files/HggPhoId_barrel_Moriond2017_wRhoRew.weights.xml");
    //photonIdMva_bkg_->BookMVA("BDTG","/afs/cern.ch/work/p/prrout/public/Myanalysis_2022/BDT_training_UL16_Gjet_Standard_analysis_25022021/Rereco_weight_files/HggPhoId_barrel_Moriond2017_wRhoRew.weights.xml");

   
  // UL18 Standard analysis TMVA train
  //photonIdMva_sig_->BookMVA("BDTG","/afs/cern.ch/work/p/prrout/public/Myanalysis_2019/BDT_training_UL18_GJet_Standard_analysis_24012021/woCorr_EB_SATrain_BDTG_UL18/weights/PhoID_barrel_UL18_GJetMC_SATrain_BDTG_nTree2k_BDTG.weights.xml");
  //photonIdMva_bkg_->BookMVA("BDTG","/afs/cern.ch/work/p/prrout/public/Myanalysis_2019/BDT_training_UL18_GJet_Standard_analysis_24012021/woCorr_EB_SATrain_BDTG_UL18/weights/PhoID_barrel_UL18_GJetMC_SATrain_BDTG_nTree2k_BDTG.weights.xml");


  // UL18 standard analysis XGB train
  //photonIdMva_sig_->BookMVA("BDT","/afs/cern.ch/work/p/prrout/public/Myanalysis_2019/BDT_training_UL18_GJet_Standard_analysis_24012021/XGBoost_train_UL18_weight_files/barrel/Barrelmodel-2021-25-01-104059_SA_phoID_UL18_woCorr.xml");
  //photonIdMva_bkg_->BookMVA("BDT","/afs/cern.ch/work/p/prrout/public/Myanalysis_2019/BDT_training_UL18_GJet_Standard_analysis_24012021/XGBoost_train_UL18_weight_files/barrel/Barrelmodel-2021-25-01-104059_SA_phoID_UL18_woCorr.xml");


  // Autumn18 standard analysis TMVA train
  //photonIdMva_sig_->BookMVA("BDTG","/afs/cern.ch/work/p/prrout/public/Myanalysis_2019/BDT_training_UL18_GJet_Standard_analysis_24012021/Autumn18_train_weight_files/PhoID_94X_barrel_RunIIAutumn18_GJetMC_SATrain_M95_BDT_woCorr_BDTG.weights.xml");
  //photonIdMva_bkg_->BookMVA("BDTG","/afs/cern.ch/work/p/prrout/public/Myanalysis_2019/BDT_training_UL18_GJet_Standard_analysis_24012021/Autumn18_train_weight_files/PhoID_94X_barrel_RunIIAutumn18_GJetMC_SATrain_M95_BDT_woCorr_BDTG.weights.xml");

  // UL17 standard analysis TMVA train
  //photonIdMva_sig_->BookMVA("BDTG","/afs/cern.ch/work/p/prrout/public/Myanalysis_2019/BDT_training_UL18_GJet_Standard_analysis_24012021/UL2017_train_weight_files/barrel/PhoID_barrel_UL2017_GJetMC_SATrain_nTree2k_LR_0p1_13052020_BDTG.weights.xml");
  //photonIdMva_bkg_->BookMVA("BDTG","/afs/cern.ch/work/p/prrout/public/Myanalysis_2019/BDT_training_UL18_GJet_Standard_analysis_24012021/UL2017_train_weight_files/barrel/PhoID_barrel_UL2017_GJetMC_SATrain_nTree2k_LR_0p1_13052020_BDTG.weights.xml");
  
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Fall17 Lowmass Train 
  //photonIdMva_sig_->BookMVA("BDT","/afs/cern.ch/work/p/prrout/public/Myanalysis_2019/BDT_Training_Summer16_Gjet_Lowmass_12062020/Fall17_IDMVA_weights_Lowmass/PhotonID_94X_barrel_woCorr_RunIIFall17_3_1_0_MCv2_Lowmass_18pT18_M55_BDT.weights.xml");
  //photonIdMva_bkg_->BookMVA("BDT","/afs/cern.ch/work/p/prrout/public/Myanalysis_2019/BDT_Training_Summer16_Gjet_Lowmass_12062020/Fall17_IDMVA_weights_Lowmass/PhotonID_94X_barrel_woCorr_RunIIFall17_3_1_0_MCv2_Lowmass_18pT18_M55_BDT.weights.xml");

  // Summer16 legacy LM train 

  //photonIdMva_sig_->BookMVA("BDTG","/afs/cern.ch/work/p/prrout/public/Myanalysis_2019/BDT_Training_Summer16_Gjet_Lowmass_12062020/woCorr_EB_LMTrain_BDTG_Summer16MC_LR_0p1_12062020/weights/PhoID_barrel_Summer16_GJetMC_LMTrain_BDTG_nTree2k_LR_0p1_12062020.root_BDTG.weights.xml");
  //photonIdMva_bkg_->BookMVA("BDTG","/afs/cern.ch/work/p/prrout/public/Myanalysis_2019/BDT_Training_Summer16_Gjet_Lowmass_12062020/woCorr_EB_LMTrain_BDTG_Summer16MC_LR_0p1_12062020/weights/PhoID_barrel_Summer16_GJetMC_LMTrain_BDTG_nTree2k_LR_0p1_12062020.root_BDTG.weights.xml");

  // UL2017 SA train Xgboost  
  //photonIdMva_sig_->BookMVA("BDT","/afs/cern.ch/work/p/prrout/public/Myanalysis_2019/BDT_Training_SA_UL2017_13052020/Xgboost_converted_TMVA_weights_UL2017/Barrelmodel-2020-17-05-000120_SA_phoID_UL2017_woDiffcorr.xml");
  //photonIdMva_bkg_->BookMVA("BDT","/afs/cern.ch/work/p/prrout/public/Myanalysis_2019/BDT_Training_SA_UL2017_13052020/Xgboost_converted_TMVA_weights_UL2017/Barrelmodel-2020-17-05-000120_SA_phoID_UL2017_woDiffcorr.xml");

  // UL2017 SA train for LR = 0.05

  //photonIdMva_sig_->BookMVA("BDTG","/afs/cern.ch/work/p/prrout/public/Myanalysis_2019/BDT_Training_SA_UL2017_13052020/woCorr_EB_SATrain_BDTG_UL2017MC_LR_0p1_13052020/weights/PhoID_barrel_UL2017_GJetMC_SATrain_BDTG_nTree2k_LR_0p1_13052020.root_BDTG.weights.xml");
  //photonIdMva_bkg_->BookMVA("BDTG","/afs/cern.ch/work/p/prrout/public/Myanalysis_2019/BDT_Training_SA_UL2017_13052020/woCorr_EB_SATrain_BDTG_UL2017MC_LR_0p1_13052020/weights/PhoID_barrel_UL2017_GJetMC_SATrain_BDTG_nTree2k_LR_0p1_13052020.root_BDTG.weights.xml");

  // Fall17 lowmass train 
  //photonIdMva_sig_->BookMVA("BDT","/afs/cern.ch/work/p/prrout/public/Myanalysis_2019/BDT_Training_Lowmass_2018_12012019/2017_Lowmass_photonID_weights/PhotonID_94X_barrel_woCorr_RunIIFall17_3_1_0_MCv2_Lowmass_18pT18_M55_BDT.weights.xml");
  //photonIdMva_bkg_->BookMVA("BDT","/afs/cern.ch/work/p/prrout/public/Myanalysis_2019/BDT_Training_Lowmass_2018_12012019/2017_Lowmass_photonID_weights/PhotonID_94X_barrel_woCorr_RunIIFall17_3_1_0_MCv2_Lowmass_18pT18_M55_BDT.weights.xml");

  // Autumn18 woCorr Lowmass
  //photonIdMva_sig_->BookMVA("BDTG","/afs/cern.ch/work/p/prrout/public/Myanalysis_2019/BDT_Training_Lowmass_2018_12012019/woCorr_PhoIso_EB_LMTrain_BDTG_RunIIAutumn18MC_LM_13012020/weights/PhoID_94X_barrel_RunIIAutumn18_GJetMC_LMTrain_M50_BDTG_nTree2k_13012020.root_BDTG.weights.xml");
  //photonIdMva_bkg_->BookMVA("BDTG","/afs/cern.ch/work/p/prrout/public/Myanalysis_2019/BDT_Training_Lowmass_2018_12012019/woCorr_PhoIso_EB_LMTrain_BDTG_RunIIAutumn18MC_LM_13012020/weights/PhoID_94X_barrel_RunIIAutumn18_GJetMC_LMTrain_M50_BDTG_nTree2k_13012020.root_BDTG.weights.xml");

// Autumn18 wocorr 
  //photonIdMva_sig_->BookMVA("BDTG","/afs/cern.ch/work/p/prrout/public/Myanalysis_2019/BDT_Training_SA_31082019/09092019_Newlytrained/woCorr_PhoIso_EB_SATrain_BDT_M95_nTree2k/weights/PhoID_94X_barrel_RunIIAutumn18_GJetMC_SATrain_M95_BDT_woCorr_BDTG.weights.xml");
  //photonIdMva_bkg_->BookMVA("BDTG","/afs/cern.ch/work/p/prrout/public/Myanalysis_2019/BDT_Training_SA_31082019/09092019_Newlytrained/woCorr_PhoIso_EB_SATrain_BDT_M95_nTree2k/weights/PhoID_94X_barrel_RunIIAutumn18_GJetMC_SATrain_M95_BDT_woCorr_BDTG.weights.xml");


  // Fall17 wocorr
  //photonIdMva_sig_->BookMVA("BDT","/afs/cern.ch/work/p/prrout/public/BDT_training_11062019/CMSSW_9_4_9/src/SA_train_wHOE_M95_13062019/woCorr_PhoIso_EB_SATrain_M95_woShowershape_BDT_wHOE_nTree2k/weights/PhoId_94X_barrel_woCorr_RunIIFall17_3_1_0_MCv2_SATrain_woShowershape_M95_BDT_wHOE_BDT.weights.xml");
  //photonIdMva_bkg_->BookMVA("BDT","/afs/cern.ch/work/p/prrout/public/BDT_training_11062019/CMSSW_9_4_9/src/SA_train_wHOE_M95_13062019/woCorr_PhoIso_EB_SATrain_M95_woShowershape_BDT_wHOE_nTree2k/weights/PhoId_94X_barrel_woCorr_RunIIFall17_3_1_0_MCv2_SATrain_woShowershape_M95_BDT_wHOE_BDT.weights.xml");

  // Autumn18 TMVA  weights (XGBTuned parameters, LR = 0.05, Max_depth = 19, nTree = 2000) 28/11/2019                                          
  //photonIdMva_sig_->BookMVA("BDTG","/afs/cern.ch/work/p/prrout/public/Myanalysis_2019/BDT_Training_SA_31082019/09092019_Newlytrained/woCorr_PhoIso_EB_SATrain_BDT_M95_xgbTuned_param_nTree2k_23112019/weights/PhoID_94X_barrel_RunIIAutumn18_GJetMC_SATrain_M95_BDT_woCorr_xgbTuned_nTree2k_23112019_BDTG.weights.xml");

  //photonIdMva_bkg_->BookMVA("BDTG","/afs/cern.ch/work/p/prrout/public/Myanalysis_2019/BDT_Training_SA_31082019/09092019_Newlytrained/woCorr_PhoIso_EB_SATrain_BDT_M95_xgbTuned_param_nTree2k_23112019/weights/PhoID_94X_barrel_RunIIAutumn18_GJetMC_SATrain_M95_BDT_woCorr_xgbTuned_nTree2k_23112019_BDTG.weights.xml");

  // Autumn18 TMVA  weights (XGBTuned parameters, LR = 0.05, Max_depth = 19, nTree = 683) 19/11/2019                                          
  //photonIdMva_sig_->BookMVA("BDTG","/afs/cern.ch/work/p/prrout/public/Myanalysis_2019/BDT_Training_SA_31082019/09092019_Newlytrained/woCorr_PhoIso_EB_SATrain_BDT_M95_xgbTuned_param_19112019/weights/PhoID_94X_barrel_RunIIAutumn18_GJetMC_SATrain_M95_BDT_woCorr_xgbTuned_19112019_BDTG.weights.xml");                                                                         
  
  //photonIdMva_bkg_->BookMVA("BDTG","/afs/cern.ch/work/p/prrout/public/Myanalysis_2019/BDT_Training_SA_31082019/09092019_Newlytrained/woCorr_PhoIso_EB_SATrain_BDT_M95_xgbTuned_param_19112019/weights/PhoID_94X_barrel_RunIIAutumn18_GJetMC_SATrain_M95_BDT_woCorr_xgbTuned_19112019_BDTG.weights.xml");  

  // Autumn18 xgboost converted TMVA weights (wTuned parameters) 19/11/2019
  //photonIdMva_sig_->BookMVA("BDT","/afs/cern.ch/work/p/prrout/public/Myanalysis_2019/BDT_Training_SA_31082019/09092019_Newlytrained/Autumn18_TMVA_xgb_withtunedparam/Barrelmodel-2019-20-11-000048_19112019.xml");                                                                                 
  //photonIdMva_bkg_->BookMVA("BDT","/afs/cern.ch/work/p/prrout/public/Myanalysis_2019/BDT_Training_SA_31082019/09092019_Newlytrained/Autumn18_TMVA_xgb_withtunedparam/Barrelmodel-2019-20-11-000048_19112019.xml");                                                                                 

  // Autumn18 xgboost wocorr                                                                                                                             
  //photonIdMva_sig_->BookMVA("BDT","/afs/cern.ch/work/p/prrout/public/Myanalysis_2019/BDT_Training_SA_31082019/09092019_Newlytrained/woCorr_xgb_barrel_weight/Barrelmodel-2019-11-09-101335_11092019.xml");
  //photonIdMva_bkg_->BookMVA("BDT","/afs/cern.ch/work/p/prrout/public/Myanalysis_2019/BDT_Training_SA_31082019/09092019_Newlytrained/woCorr_xgb_barrel_weight/Barrelmodel-2019-11-09-101335_11092019.xml");


  //extended phase space by prasant wHOE on BDTG
  //photonIdMva_sig_->BookMVA( "BDTG", "/eos/user/p/prrout/public/Flashgg/PhotonIDMVA/Extended_Space_05062019/woCorr_PhoIso_EB_extendSpace_M55_woShowershape_BDTG_wHOE_nTree2k/weights/PhoId_94X_barrel_woCorr_RunIIFall17_3_1_0_MCv2_extendspace_woShowershape_M55_BDTG_BDTG.weights.xml");
  //photonIdMva_bkg_->BookMVA( "BDTG", "/eos/user/p/prrout/public/Flashgg/PhotonIDMVA/Extended_Space_05062019/woCorr_PhoIso_EB_extendSpace_M55_woShowershape_BDTG_wHOE_nTree2k/weights/PhoId_94X_barrel_woCorr_RunIIFall17_3_1_0_MCv2_extendspace_woShowershape_M55_BDTG_BDTG.weights.xml");

  // standard analysis by prasant wHOE on BDT
  //photonIdMva_sig_->BookMVA("BDT","/afs/cern.ch/work/p/prrout/public/BDT_training_11062019/CMSSW_9_4_9/src/SA_train_wHOE_M95_13062019/woCorr_PhoIso_EB_SATrain_M95_woShowershape_BDT_wHOE_nTree2k/weights/PhoId_94X_barrel_woCorr_RunIIFall17_3_1_0_MCv2_SATrain_woShowershape_M95_BDT_wHOE_BDT.weights.xml");
  //photonIdMva_bkg_->BookMVA("BDT","/afs/cern.ch/work/p/prrout/public/BDT_training_11062019/CMSSW_9_4_9/src/SA_train_wHOE_M95_13062019/woCorr_PhoIso_EB_SATrain_M95_woShowershape_BDT_wHOE_nTree2k/weights/PhoId_94X_barrel_woCorr_RunIIFall17_3_1_0_MCv2_SATrain_woShowershape_M95_BDT_wHOE_BDT.weights.xml");

  //standard analysis by kuntal woHOE on BDT
  //photonIdMva_sig_->BookMVA( "BDT", "/eos/user/p/prrout/public/Flashgg/PhotonIDMVA/Extended_Space_05062019/standard_weights_kuntal/HggPhoId_94X_barrel_BDT_v2.weights.xml");
  //photonIdMva_bkg_->BookMVA( "BDT", "/eos/user/p/prrout/public/Flashgg/PhotonIDMVA/Extended_Space_05062019/standard_weights_kuntal/HggPhoId_94X_barrel_BDT_v2.weights.xml");
  
  // low mass analysis by prasant wHOE on BDT
  //photonIdMva_sig_->BookMVA( "BDT", "/eos/user/p/prrout/public/Flashgg/PhotonIDMVA/Extended_Space_05062019/Lowmass_weights_prasant/PhoId_94X_barrel_woCorr_RunIIFall17_3_1_0_MCv2_Lowmass_18pT18_M55_BDT.weights.xml");
  //photonIdMva_bkg_->BookMVA( "BDT", "/eos/user/p/prrout/public/Flashgg/PhotonIDMVA/Extended_Space_05062019/Lowmass_weights_prasant/PhoId_94X_barrel_woCorr_RunIIFall17_3_1_0_MCv2_Lowmass_18pT18_M55_BDT.weights.xml");

  // standard analysis by prasant woHOE on BDTG
  //photonIdMva_sig_->BookMVA( "BDTG", "/eos/user/p/prrout/public/Flashgg/PhotonIDMVA/stanalysis/PhotonIDMVA_StAnalysis_06102018/woCorr_PhoIso_EB_Stdmass_M95_woShowershape_BDTG_nTree2k/weights/PhoId_94X_barrel_woCorr_RunIIFall17_3_1_0_MCv2_Stdmass_woShowershape_BDTG_M95_BDTG.weights.xml");
  //photonIdMva_bkg_->BookMVA( "BDTG", "/eos/user/p/prrout/public/Flashgg/PhotonIDMVA/stanalysis/PhotonIDMVA_StAnalysis_06102018/woCorr_PhoIso_EB_Stdmass_M95_woShowershape_BDTG_nTree2k/weights/PhoId_94X_barrel_woCorr_RunIIFall17_3_1_0_MCv2_Stdmass_woShowershape_BDTG_M95_BDTG.weights.xml");

  for(int i = 0; i < nEntries_sig; i++){
    t_sig->GetEntry(i);
    out_mvares_s = photonIdMva_sig_->EvaluateMVA( "BDTG" );
    if(abs(out_scEta_s)<1.5){ 
      outputTree_s->Fill();
      //cout << "BDT score EB signal =" << out_mvares_s <<  endl;
    }
  }

  cout <<"signal evaluated" << endl;

  for(int i = 0; i < nEntries_bkg; i++){
    t_bkg->GetEntry(i);
    out_mvares_b = photonIdMva_bkg_->EvaluateMVA( "BDTG" );
    if(abs(out_scEta_b)<1.5){
      outputTree_b->Fill();
      //cout << "BDT score EB bkg =" << out_mvares_b <<  endl;
    }
  }

  cout <<"background evaluated" << endl;

  outputFile->Write();

}

