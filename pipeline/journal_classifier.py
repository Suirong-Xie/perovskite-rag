#!/usr/bin/env python3
"""
journal_classifier.py — 共享的期刊名 → 目录名分类器

被 campus_download.py / s2_download_pdfs.py / audit_classification.py 共用。

设计原则:
  1. 精确匹配优先 (规范化后)
  2. 最长 key 优先子串匹配 (防止 "science" 吞噬 "Science Advances" 等)
  3. HTML 实体解码 (&amp; → &)
  4. 去掉括号注释 ("Nano letters (Print)" → "Nano letters")
  5. 无匹配时: 安全化 venue 名, 不再截断

用法:
  from pipeline.journal_classifier import classify_venue

  dirname = classify_venue("ACS Energy Letters")  # → "ACS_Energy_Letters"
  dirname = classify_venue("")                     # → "Unknown"
"""

import re
import html as _html


# ── 期刊名 → 安全目录名 (精确匹配) ──

EXACT_VENUE_MAP: dict[str, str] = {
    # ── Nature 家族 ──
    "Nature": "Nature",
    "Nature Energy": "NatEnergy",
    "Nature Materials": "NatMater",
    "Nature Photonics": "NatPhoton",
    "Nature Nanotechnology": "NatNanotech",
    "Nature Communications": "NatComm",
    "Nature Reviews Materials": "NatRevMater",
    "Nature Synthesis": "NatSynthesis",
    "Nature Sustainability": "NatSustainability",
    "Nature Reviews Chemistry": "NatRevChemistry",
    "Nature Chemistry": "NatChemistry",
    "Nature Physics": "NatPhysics",
    "Nature Electronics": "NatElectronics",
    "Nature Catalysis": "NatCatalysis",
    "Nature Computational Science": "NatComputSci",
    "Nature Reviews Physics": "NatRevPhysics",
    "Nature Reviews Clean Technology": "NatRevCleanTech",
    "Nature Reviews Methods Primers": "NatRevMethodsPrimers",
    "Nature Reviews Electrical Engineering": "NatRevElectricalEng",
    "Nature Protocols": "NatProtocols",
    "Nature Sensors": "Nature_Sensors",
    "Nature Reviews Materials (Print)": "NatRevMater",
    "Nature Energy (Print)": "NatEnergy",
    "Nature Communications (Print)": "NatComm",
    "Nature Photonics (Print)": "NatPhoton",
    "Nature Nanotechnology (Print)": "NatNanotech",
    "Nature Materials (Print)": "NatMater",
    "Nature Chemistry (Print)": "NatChemistry",
    "Nature Physics (Print)": "NatPhysics",
    "Nature Electronics (Print)": "NatElectronics",
    # 缩写形式
    "Nat Energy": "NatEnergy",
    "Nat. Energy": "NatEnergy",
    "Nat Mater": "NatMater",
    "Nat. Mater.": "NatMater",
    "Nat Photonics": "NatPhoton",
    "Nat. Photonics": "NatPhoton",
    "Nat Nanotechnol": "NatNanotech",
    "Nat. Nanotechnol.": "NatNanotech",
    "Nat Nanotechnology": "NatNanotech",
    "Nat. Nanotechnology": "NatNanotech",
    "Nat Commun": "NatComm",
    "Nat. Commun.": "NatComm",
    "Nat Chemistry": "NatChemistry",
    "Nat. Chem.": "NatChemistry",
    "Nat Physics": "NatPhysics",
    "Nat. Phys.": "NatPhysics",
    "Nat Electronics": "NatElectronics",
    "Nat. Electron.": "NatElectronics",
    "Nat Catalysis": "NatCatalysis",
    "Nat. Catal.": "NatCatalysis",
    "Nat Synthesis": "NatSynthesis",
    "Nat. Synth.": "NatSynthesis",
    "Nat Sustainability": "NatSustainability",
    "Nat. Sustain.": "NatSustainability",
    "Nat Rev Mater": "NatRevMater",
    "Nat. Rev. Mater.": "NatRevMater",
    "Nat Rev Chem": "NatRevChemistry",
    "Nat. Rev. Chem.": "NatRevChemistry",
    "Nat Rev Phys": "NatRevPhysics",
    "Nat. Rev. Phys.": "NatRevPhysics",
    "Nat Comput Sci": "NatComputSci",
    "Nat. Comput. Sci.": "NatComputSci",
    # Nature 不带空格 (DOI metadata 有时连写)
    "NatureEnergy": "NatEnergy",
    "NatureMaterials": "NatMater",
    "NaturePhotonics": "NatPhoton",
    "NatureNanotechnology": "NatNanotech",
    "NatureCommunications": "NatComm",
    "NatureChemistry": "NatChemistry",
    "NaturePhysics": "NatPhysics",
    "NatureElectronics": "NatElectronics",
    "NatureCatalysis": "NatCatalysis",
    "NatureSynthesis": "NatSynthesis",
    "NatureSustainability": "NatSustainability",
    # 带 Nature Publishing Group 标注
    "Nature Energy (Nature Publishing Group)": "NatEnergy",
    "Nature Communications (Nature Publishing Group)": "NatComm",
    "Nature Materials (Nature Publishing Group)": "NatMater",

    # ── Science 家族 ──
    "Science": "Science",
    "Science Advances": "Science_Advances",
    "Science Immunology": "Science_Immunology",
    "Science Robotics": "Science_Robotics",

    # ── Cell / Joule / Chem / Matter ──
    "Joule": "Joule",
    "Chem": "Chem",
    "Matter": "Matter",
    "Cell Reports Physical Science": "Cell_Reports_Physical_Science",
    "iScience": "iScience",
    "Trends in Chemistry": "Trends_in_Chemistry",
    "DeCarbon": "DeCarbon",
    "The Innovation": "The_Innovation",
    "Innovation (Cambridge (Mass.))": "The_Innovation",
    "Innovation (Cambridge (Mass))": "The_Innovation",

    # ── ACS ──
    "ACS Energy Letters": "ACS_Energy_Letters",
    "ACS Applied Materials & Interfaces": "ACS_Applied_Materials_Interfaces",
    "ACS Applied Materials and Interfaces": "ACS_Applied_Materials_Interfaces",
    "ACS Nano": "ACS_Nano",
    "Nano Letters": "Nano_Letters",
    "Journal of the American Chemical Society": "JACS",
    "Chemistry of Materials": "Chemistry_of_Materials",
    "ACS Applied Energy Materials": "ACS_Applied_Energy_Materials",
    "ACS Omega": "ACS_Omega",
    "Journal of Physical Chemistry Letters": "J_Physical_Chemistry_Letters",
    "The Journal of Physical Chemistry Letters": "J_Physical_Chemistry_Letters",
    "ACS Photonics": "ACS_Photonics",
    "ACS Central Science": "ACS_Central_Science",
    "Chemical Reviews": "Chemical_Reviews",
    "Accounts of Chemical Research": "Accounts_of_Chemical_Research",
    "Inorganic Chemistry": "Inorganic_Chemistry",
    "The Journal of Physical Chemistry C": "J_Physical_Chemistry_C",
    "Journal of Physical Chemistry C": "J_Physical_Chemistry_C",
    "Journal of Chemical & Engineering Data": "J_Chemical_Engineering_Data",

    # ── RSC ──
    "Energy & Environmental Science": "Energy_Environmental_Science",
    "Energy &amp; Environmental Science": "Energy_Environmental_Science",
    "Journal of Materials Chemistry A": "J_Materials_Chemistry_A",
    "Journal of Materials Chemistry C": "J_Materials_Chemistry_C",
    "Nanoscale": "Nanoscale",
    "Chemical Science": "Chemical_Science",
    "Chemical Communications": "Chemical_Communications",
    "Chemical Society Reviews": "Chemical_Society_Reviews",
    "RSC Advances": "RSC_Advances",
    "Materials Horizons": "Materials_Horizons",
    "Physical Chemistry Chemical Physics": "PCCP",
    "Physical Chemistry, Chemical Physics - PCCP": "PCCP",
    "EES Solar": "EES_Solar",
    "Materials Advances": "Materials_Advances",
    "Nanoscale Advances": "Nanoscale_Advances",
    "Nanoscale Horizons": "Nanoscale_Horizons",

    # ── Wiley / Angewandte 等 ──
    "Angewandte Chemie": "Angewandte_Chemie",
    "Advanced Materials": "Advanced_Materials",
    "Advanced Energy Materials": "Advanced_Energy_Materials",
    "Advanced Functional Materials": "Advanced_Functional_Materials",
    "Advanced Science": "Advanced_Science",
    "Small": "Small",
    "Small Methods": "Small_Methods",
    "Solar RRL": "Solar_RRL",
    "Advanced Optical Materials": "Advanced_Optical_Materials",
    "Advanced Electronic Materials": "Advanced_Electronic_Materials",
    "Laser & Photonics Reviews": "Laser_Photonics_Reviews",
    "Advanced Materials Interfaces": "Advanced_Materials_Interfaces",
    "Advanced Materials Technologies": "Advanced_Materials_Technologies",
    "InfoMat": "InfoMat",
    "EcoMat": "EcoMat",
    "SusMat": "SusMat",
    "Carbon Energy": "Carbon_Energy",

    # ── Elsevier ──
    "Nano Energy": "Nano_Energy",
    "Chemical Engineering Journal": "Chemical_Engineering_Journal",
    "Solar Energy": "Solar_Energy",
    "Solar Energy Materials and Solar Cells": "Solar_Energy_Materials_Solar_Cells",
    "Solar Energy Materials & Solar Cells": "Solar_Energy_Materials_Solar_Cells",
    "Nano Today": "Nano_Today",
    "Journal of Energy Chemistry": "Journal_of_Energy_Chemistry",
    "Journal of Colloid and Interface Science": "J_Colloid_Interface_Science",
    "Journal of Physics and Chemistry of Solids": "J_Physics_Chemistry_of_Solids",
    "Applied Surface Science": "Applied_Surface_Science",
    "Materials Today": "Materials_Today",
    "Materials Today Physics": "Materials_Today_Physics",
    "Materials Today Energy": "Materials_Today_Energy",
    "Materials Today Proceedings": "Materials_Today_Proceedings",
    "Materials Today: Proceedings": "Materials_Today_Proceedings",
    "Renewable and Sustainable Energy Reviews": "Renewable_Sustainable_Energy_Reviews",
    "Renewable & Sustainable Energy Reviews": "Renewable_Sustainable_Energy_Reviews",
    "Sustainable Energy Technologies and Assessments": "Sustainable_Energy_Technologies_Assessments",
    "Journal of Hazardous Materials": "J_Hazardous_Materials",
    "Computational Materials Science": "Computational_Materials_Science",
    "Electrochimica Acta": "Electrochimica_Acta",
    "Materials Chemistry and Physics": "Materials_Chemistry_and_Physics",
    "Materials Research Bulletin": "Materials_Research_Bulletin",
    "Materials research bulletin": "Materials_Research_Bulletin",
    "Results in Engineering": "Results_in_Engineering",
    "Energy &amp; Fuels": "Energy_Fuels",
    "Energy & Fuels": "Energy_Fuels",
    "Energy Reports": "Energy_Reports",
    "Solid State Communications": "Solid_State_Communications",
    "Surface and Coatings Technology": "Surface_Coatings_Technology",
    "Scripta Materialia": "Scripta_Materialia",
    "Ceramics International": "Ceramics_International",
    "Thin Solid Films": "Thin_Solid_Films",
    "Optical Materials": "Optical_Materials",
    "Optical materials (Amsterdam)": "Optical_Materials",
    "Optical materials": "Optical_Materials",
    "Optik": "Optik",
    "Optik (Stuttgart)": "Optik",
    "Journal of Alloys and Compounds": "J_Alloys_and_Compounds",
    "Materials Science and Engineering B": "Mater_Science_Engineering_B",
    "Materials Science and Engineering: B": "Mater_Science_Engineering_B",
    "Materials Science in Semiconductor Processing": "Mater_Science_Semiconductor_Processing",
    "Materials Science for Energy Technologies": "Mater_Science_Energy_Technologies",
    "Journal of Power Sources": "J_Power_Sources",
    "Renewable Energy": "Renewable_Energy",
    "Energy Conversion and Management": "Energy_Conversion_Management",
    "Journal of Cleaner Production": "J_Cleaner_Production",
    "Applied Energy": "Applied_Energy",
    "International Journal of Hydrogen Energy": "Int_J_Hydrogen_Energy",
    "Materials Letters": "Materials_Letters",
    "Journal of Luminescence": "J_Luminescence",
    "Computational and Theoretical Chemistry": "Computational_Theoretical_Chemistry",
    "Chemical Physics Letters": "Chemical_Physics_Letters",

    # ── IOP ──
    "Physica Scripta": "Physica_Scripta",
    "Nanotechnology": "Nanotechnology",
    "Journal of Physics D: Applied Physics": "J_Physics_D_Applied_Physics",
    "Journal of Physics D Applied Physics": "J_Physics_D_Applied_Physics",
    "Journal of Physics: Condensed Matter": "J_Physics_Condensed_Matter",
    "Journal of Physics Condensed Matter": "J_Physics_Condensed_Matter",
    "Journal of Physics: Energy": "J_Physics_Energy",
    "Journal of Physics Energy": "J_Physics_Energy",
    "Journal of Physics: Materials": "J_Physics_Materials",
    "Journal of Physics Materials": "J_Physics_Materials",
    "Journal of Physics: Photonics": "J_Physics_Photonics",
    "Journal of Physics: Conference Series": "J_Physics_Conference_Series",
    "Journal of Optics": "J_Optics",
    "Journal of Semiconductors": "J_Semiconductors",
    "Materials Futures": "Materials_Futures",
    "Materials Research Express": "Materials_Research_Express",
    "Engineering Research Express": "Engineering_Research_Express",
    "Nano Express": "Nano_Express",
    "Nano Futures": "Nano_Futures",
    "Reports on Progress in Physics": "Reports_on_Progress_in_Physics",
    "Journal of Micromechanics and Microengineering": "J_Micromechanics_Microengineering",
    "International Journal of Extreme Manufacturing": "Int_J_Extreme_Manufacturing",
    "Semiconductor Science and Technology": "Semiconductor_Science_Technology",
    "Smart Materials and Structures": "Smart_Materials_Structures",
    "2D Materials": "2D_Materials",
    "Methods and Applications in Fluorescence": "Methods_Applications_Fluorescence",
    "Plasma Sources Science and Technology": "Plasma_Sources_Science_Technology",
    "Measurement Science and Technology": "Measurement_Science_Technology",
    "Chinese Physics B": "Chinese_Physics_B",
    "Chinese Physics Letters": "Chinese_Physics_Letters",
    "Progress in Energy": "Progress_in_Energy",
    "Flexible and Printed Electronics": "Flexible_Printed_Electronics",
    "Materials for Renewable and Sustainable Energy": "Materials_Renewable_Sustainable_Energy",
    "Surface Topography: Metrology and Properties": "Surface_Topography_Metrology",
    "Materials Research Letters": "Materials_Research_Letters",
    "Electronic Structure": "Electronic_Structure",
    "Journal of Neural Engineering": "J_Neural_Engineering",

    # ── AIP ──
    "Applied Physics Letters": "Applied_Physics_Letters",
    "AIP Advances": "AIP_Advances",
    "Journal of Applied Physics": "J_Applied_Physics",
    "APL Materials": "APL_Materials",
    "Applied Physics Reviews": "Applied_Physics_Reviews",
    "Journal of Chemical Physics": "J_Chemical_Physics",
    "The Journal of Chemical Physics": "J_Chemical_Physics",
    "Journal of Renewable and Sustainable Energy": "J_Renewable_Sustainable_Energy",

    # ── APS ──
    "Physical Review B": "Physical_Review_B",
    "Physical Review Letters": "Physical_Review_Letters",
    "Physical Review Materials": "Physical_Review_Materials",
    "Physical Review X": "Physical_Review_X",
    "Physical Review Applied": "Physical_Review_Applied",
    "Physical Review Research": "Physical_Review_Research",
    "Physical Review E": "Physical_Review_E",
    "Reviews of Modern Physics": "Reviews_of_Modern_Physics",

    # ── Springer / Nature Portfolio ──
    "Scientific Reports": "Scientific_Reports",
    "Scientific Data": "Scientific_Data",
    "Communications Materials": "Communications_Materials",
    "Communications Physics": "Communications_Physics",
    "Communications Chemistry": "Communications_Chemistry",
    "Communications Earth & Environment": "Communications_Earth_Environment",
    "Light: Science & Applications": "Light_Science_Applications",
    "Light: Science and Applications": "Light_Science_Applications",
    "Nano-Micro Letters": "Nano_Micro_Letters",
    "npj Flexible Electronics": "npj_Flexible_Electronics",
    "npj Nanophotonics": "npj_Nanophotonics",
    "npj Computational Materials": "npj_Computational_Materials",
    "npj 2D Materials and Applications": "npj_2D_Materials_Applications",
    "npj Quantum Materials": "npj_Quantum_Materials",
    "npj Quantum Information": "npj_Quantum_Information",
    "Journal of Materials Science": "J_Materials_Science",
    "Journal of Materials Science: Materials in Electronics": "J_Materials_Science_Mater_Electron",
    "Journal of Electronic Materials": "J_Electronic_Materials",
    "Journal of Inorganic and Organometallic Polymers and Materials": "J_Inorg_Organomet_Polym_Mater",

    # ── IEEE ──
    "IEEE Journal of Photovoltaics": "IEEE_J_Photovoltaics",
    "IEEE Transactions on Electron Devices": "IEEE_Trans_Electron_Devices",
    "IEEE Access": "IEEE_Access",
    "IEEE Sensors Journal": "IEEE_Sensors_Journal",

    # ── MDPI ──
    "Nanomaterials": "Nanomaterials",
    "Materials": "Materials",
    "Energies": "Energies",
    "Molecules": "Molecules",
    "Crystals": "Crystals",
    "Polymers": "Polymers",
    "Micromachines": "Micromachines",
    "Sensors": "Sensors",
    "Coatings": "Coatings",
    "Photonics": "Photonics",
    "Batteries": "Batteries",
    "Membranes": "Membranes",
    "Catalysts": "Catalysts",
    "Applied Sciences": "Applied_Sciences",
    "Chemistry": "Chemistry",
    "Inorganics": "Inorganics",
    "Magnetochemistry": "Magnetochemistry",
    "Symmetry": "Symmetry",
    "Electronics": "Electronics",
    "Chemosensors": "Chemosensors",
    "Solar": "Solar",

    # ── 其他常见 ──
    "Heliyon": "Heliyon",
    "PNAS": "PNAS",
    "Proceedings of the National Academy of Sciences": "PNAS",
    "PLOS ONE": "PLOS_ONE",
    "Advanced Photonics Research": "Advanced_Photonics_Research",
    "Advanced Photonics": "Advanced_Photonics",
    "Progress in Photovoltaics": "Progress_in_Photovoltaics",
    "Optics Express": "Optics_Express",
    "Optics Letters": "Optics_Letters",
    "Optics and Laser Technology": "Optics_Laser_Technology",
    "Optoelectronics and Advanced Materials, Rapid Communications": "Optoelectronics_Adv_Mater",
    "Journal of Materials Research": "J_Materials_Research",
    "Journal of Materials Research and Technology": "J_Mater_Research_Technology",
    "Science and Technology of Advanced Materials": "Sci_Technology_Adv_Materials",
    "Science China Materials": "Science_China_Materials",
    "IOP Conference Series: Materials Science and Engineering": "IOP_Conf_Series_Mater_Sci_Eng",
    "IOP Conference Series Materials Science and Engineering": "IOP_Conf_Series_Mater_Sci_Eng",
    "Dalton Transactions": "Dalton_Transactions",
    "CrystEngComm": "CrystEngComm",
    "Nanoscale Research Letters": "Nanoscale_Research_Letters",
    "Nanotechnology Reviews": "Nanotechnology_Reviews",
    "Journal of Physical Chemistry A": "J_Physical_Chemistry_A",
    "Journal of Physical Chemistry B": "J_Physical_Chemistry_B",
    "Langmuir": "Langmuir",
    "Nano Research": "Nano_Research",
    "Nano Reseach": "Nano_Research",  # S2 metadata 拼写错误
    "New Journal of Chemistry": "New_J_Chemistry",
    "Science Bulletin": "Science_Bulletin",
    "National Science Review": "National_Science_Review",
    "Materials Chemistry Frontiers": "Materials_Chemistry_Frontiers",
    "Materials Today Communications": "Materials_Today_Communications",
    "Materials Today Electronics": "Materials_Today_Electronics",
    "Materials Today Sustainability": "Materials_Today_Sustainability",
    "ACS Applied Electronic Materials": "ACS_Applied_Electronic_Materials",
    "ACS Applied Nano Materials": "ACS_Applied_Nano_Materials",
    "ACS Applied Bio Materials": "ACS_Applied_Bio_Materials",
    "ECS Journal of Solid State Science and Technology": "ECS_J_Solid_State_Sci_Technol",
    "Applied Physics A": "Applied_Physics_A",
    "Japanese Journal of Applied Physics": "Jpn_J_Applied_Physics",
    "Energy Advances": "Energy_Advances",
    "RSC Sustainability": "RSC_Sustainability",
    "Emergent Materials": "Emergent_Materials",
    "Accounts of Materials Research": "Accounts_of_Materials_Research",
    "Inorganic Chemistry Frontiers": "Inorganic_Chemistry_Frontiers",
    "Green Chemistry": "Green_Chemistry",
    "Analytical Chemistry": "Analytical_Chemistry",
    "Pure and Applied Chemistry": "Pure_Applied_Chemistry",
    "ACS Sustainable Chemistry &amp; Engineering": "ACS_Sustainable_Chemistry_Engineering",
    "ACS Sustainable Chemistry & Engineering": "ACS_Sustainable_Chemistry_Engineering",
    "Chemistry of Inorganic Materials": "Chemistry_of_Inorganic_Materials",
    "Science China Chemistry": "Science_China_Chemistry",
    "Science China Materials": "Science_China_Materials",
    "Science China Physics Mechanics and Astronomy": "Science_China_Phys_Mech_Astron",
    "Soft Science": "Soft_Science",
    "Applications of Surface Science": "Applications_of_Surface_Science",
    "Advances in Materials Science and Engineering": "Adv_Mater_Sci_Eng",
    "Environmental science and pollution research international": "Environ_Sci_Pollut_Res",
    "The Arabian journal for science and engineering": "Arabian_J_Sci_Eng",
    "Journal of Materials Science: Materials in Electronics": "J_Materials_Science_Mater_Electron",
    "Journal of materials science. Materials in electronics": "J_Materials_Science_Mater_Electron",
    "Royal Society Open Science": "Royal_Soc_Open_Science",
    "International Journal of Molecular Sciences": "Int_J_Molecular_Sciences",
    "Bulletin of Materials Science": "Bulletin_of_Materials_Science",
    "Journal of Sol-Gel Science and Technology": "J_Sol_Gel_Sci_Technol",
    "Science of the Total Environment": "Sci_Total_Environment",
    "Journal of Computational Electronics": "J_Computational_Electronics",
    "Results in Physics": "Results_in_Physics",
    "Computational Condensed Matter": "Computational_Condensed_Matter",
    "Advanced Composites and Hybrid Materials": "Adv_Composites_Hybrid_Mater",
    "Modelling and Simulation in Materials Science and Engineering": "Modell_Simul_Mater_Sci_Eng",
    "Electronic Materials Letters": "Electronic_Materials_Letters",
    "Journal of Materiomics": "J_Materiomics",
    "Energy Material Advances": "Energy_Material_Advances",
    "Energy Materials": "Energy_Materials",
    "ES Materials & Manufacturing": "ES_Materials_Manufacturing",
    "Materials Reports: Energy": "Materials_Reports_Energy",
    "Materials science & engineering. R, Reports": "Mater_Sci_Eng_R_Reports",
    "Transactions on Electrical and Electronic Materials": "Trans_Electrical_Electronic_Mater",
    "Emerging Materials Research": "Emerging_Materials_Research",
    "Sustainable Materials and Technologies": "Sustainable_Materials_Technologies",
    "International Journal of Minerals, Metallurgy, and Materials": "Int_J_Minerals_Metall_Mater",
    "Advances in Condensed Matter Physics": "Adv_Condensed_Matter_Physics",
    "IEEE Journal of Quantum Electronics": "IEEE_J_Quantum_Electronics",
    "ACS Sensors": "ACS_Sensors",
    "Journal of Solid State Chemistry": "J_Solid_State_Chemistry",
    "The journal of physical chemistry. C, Nanomaterials and interfaces": "J_Physical_Chemistry_C",
    "Advanced Electronic Materials": "Advanced_Electronic_Materials",
    "ChemSusChem": "ChemSusChem",
    "ChemElectroChem": "ChemElectroChem",
    "Chemistry - A European Journal": "Chemistry_A_European_Journal",
    "Chemistry - An Asian Journal": "Chemistry_An_Asian_Journal",
    "ChemistrySelect": "ChemistrySelect",
    "ChemPhotoChem": "ChemPhotoChem",
    "ChemPhysChem": "ChemPhysChem",
    "Batteries & Supercaps": "Batteries_Supercaps",
    "Energy Technology": "Energy_Technology",
    "Energy Storage Materials": "Energy_Storage_Materials",
    "Materials Today Chemistry": "Materials_Today_Chemistry",
    "Sustainable Energy & Fuels": "Sustainable_Energy_Fuels",
    "Sustainable Energy and Fuels": "Sustainable_Energy_Fuels",
    "Journal of Chemical Theory and Computation": "J_Chemical_Theory_Computation",
    "The Journal of Chemical Theory and Computation": "J_Chemical_Theory_Computation",
    "Journal of Computational Chemistry": "J_Computational_Chemistry",
    "WIREs Computational Molecular Science": "WIREs_Comput_Mol_Science",
    "Journal of Chemical Information and Modeling": "J_Chemical_Info_Modeling",
    "NPG Asia Materials": "NPG_Asia_Materials",
    "APL Energy": "APL_Energy",
    "PRX Energy": "PRX_Energy",
    "ACS Materials Letters": "ACS_Materials_Letters",
    "ACS Materials Au": "ACS_Materials_Au",
    "ACS Physical Chemistry Au": "ACS_Physical_Chemistry_Au",
    "JACS Au": "JACS_Au",
    "PhotoniX": "PhotoniX",
    "eScience": "eScience",
    "InfoScience": "InfoScience",
    "Interdisciplinary Materials": "Interdisciplinary_Materials",
    "SmartMat": "SmartMat",
    "FlexMat": "FlexMat",
    "Responsive Materials": "Responsive_Materials",
    "Wearable Electronics": "Wearable_Electronics",
    "Next Materials": "Next_Materials",
    "Next Energy": "Next_Energy",
    "Next Nanotechnology": "Next_Nanotechnology",
    "Next Sustainability": "Next_Sustainability",
    "Device": "Device",
    "Chem Catalysis": "Chem_Catalysis",
    "JPhys Energy": "JPhys_Energy",
    "JPhys Materials": "JPhys_Materials",
    "JPhys Photonics": "JPhys_Photonics",
    "JPhysD: Applied Physics": "J_Physics_D_Applied_Physics",
    "JPhys Complexity": "JPhys_Complexity",
    "Progress in Materials Science": "Progress_in_Materials_Science",
    "Progress in Polymer Science": "Progress_in_Polymer_Science",
    "Progress in Quantum Electronics": "Progress_in_Quantum_Electronics",
    "Reports on Progress in Physics (Print)": "Reports_on_Progress_in_Physics",
    "Materials Science and Engineering: R: Reports": "Mater_Sci_Eng_R_Reports",
    "Chemical Society Reviews (Print)": "Chemical_Society_Reviews",
    "Optica": "Optica",
    "Optics Communications": "Optics_Communications",
    "Optics & Laser Technology": "Optics_Laser_Technology",
    "Optical and Quantum Electronics": "Optical_Quantum_Electronics",
    "Infrared Physics & Technology": "Infrared_Physics_Technology",
    "Spectrochimica Acta Part A: Molecular and Biomolecular Spectroscopy": "Spectrochimica_Acta_A",
    "Journal of Raman Spectroscopy": "J_Raman_Spectroscopy",
    "Journal of Molecular Structure": "J_Molecular_Structure",
    "Coordination Chemistry Reviews": "Coordination_Chemistry_Reviews",
    "Advances in Colloid and Interface Science": "Advances_in_Colloid_Interface_Science",
    "Current Opinion in Solid State and Materials Science": "Curr_Opinion_Solid_State_Mater_Sci",
    "Chemical Engineering Science": "Chemical_Engineering_Science",
    "Separation and Purification Technology": "Separation_Purification_Technology",
    "Fuel": "Fuel",
    "Carbon": "Carbon",
    "Diamond and Related Materials": "Diamond_Related_Materials",
    "Surfaces and Interfaces": "Surfaces_and_Interfaces",
    "Advanced Materials Interfaces (Print)": "Advanced_Materials_Interfaces",
    "Physics Reports": "Physics_Reports",
    "Physics Letters A": "Physics_Letters_A",
    "Physics Letters B": "Physics_Letters_B",
    "Modern physics letters B": "Modern_Physics_Letters_B",
    "Modern Physics Letters B": "Modern_Physics_Letters_B",
    "International Journal of Modern Physics B": "Int_J_Modern_Physics_B",
    "Physics Today": "Physics_Today",
    "Reviews of Modern Plasma Physics": "Rev_Modern_Plasma_Physics",
    "New Journal of Physics": "New_J_Physics",
    "European Physical Journal Plus": "Eur_Physical_J_Plus",
    "European Physical Journal B": "Eur_Physical_J_B",
    "European Physical Journal C": "Eur_Physical_J_C",
    "Europhysics Letters": "Europhysics_Letters",
    "Physica B: Condensed Matter": "Physica_B_Condensed_Matter",
    "Physica. B, Condensed matter": "Physica_B_Condensed_Matter",
    "Physica B": "Physica_B",
    "Physica E: Low-dimensional Systems and Nanostructures": "Physica_E",
    "Physica Status Solidi A": "Physica_Status_Solidi_A",
    "Physica Status Solidi B": "Physica_Status_Solidi_B",
    "Physica Status Solidi (RRL)": "Physica_Status_Solidi_RRL",
    "Physica Status Solidi RRL": "Physica_Status_Solidi_RRL",
    "Physical Chemistry Chemical Physics (PCCP)": "PCCP",
    "Faraday Discussions": "Faraday_Discussions",
    "Bulletin of the Chemical Society of Japan": "Bull_Chem_Soc_Japan",
    "Chemical Physics": "Chemical_Physics",
    "Chemical Physics Impact": "Chemical_Physics_Impact",
    "Chemical Engineering and Processing": "Chemical_Engineering_Processing",
    "Journal of Photochemistry and Photobiology A: Chemistry": "J_Photochem_Photobiol_A",
    "Journal of Photochemistry and Photobiology C: Photochemistry Reviews": "J_Photochem_Photobiol_C",
    "Journal of Photochemistry & Photobiology, A: Chemistry": "J_Photochem_Photobiol_A",
    "Journal of Photochemistry and Photobiology": "J_Photochem_Photobiol",
    "Dyes and Pigments": "Dyes_and_Pigments",
    "Synthetic Metals": "Synthetic_Metals",
    "Organic Electronics": "Organic_Electronics",
    "Organic Electronics (Print)": "Organic_Electronics",
    "Polymer": "Polymer",
    "Polymer Chemistry": "Polymer_Chemistry",
    "Polymer Testing": "Polymer_Testing",
    "Polymer Composites": "Polymer_Composites",
    "Composites Part B: Engineering": "Composites_Part_B_Engineering",
    "Composites Science and Technology": "Composites_Science_Technology",
    "Composites Part A: Applied Science and Manufacturing": "Composites_Part_A",
    "Construction and Building Materials": "Construction_Building_Materials",
    "Cement and Concrete Composites": "Cement_Concrete_Composites",
    "Corrosion Science": "Corrosion_Science",
    "Tribology International": "Tribology_International",
    "Wear": "Wear",
    "Vacuum": "Vacuum",
    "Journal of Vacuum Science & Technology A": "J_Vac_Sci_Technology_A",
    "Journal of Vacuum Science & Technology B": "J_Vac_Sci_Technology_B",
    "Journal of Vacuum Science and Technology A": "J_Vac_Sci_Technology_A",
    "Microelectronic Engineering": "Microelectronic_Engineering",
    "Microelectronics Reliability": "Microelectronics_Reliability",
    "Microelectronics Journal": "Microelectronics_Journal",
    "Solid-State Electronics": "Solid_State_Electronics",
    "IEEE Electron Device Letters": "IEEE_Electron_Device_Letters",
    "IEEE Journal of Selected Topics in Quantum Electronics": "IEEE_J_Sel_Topics_Quantum_Electron",
    "IEEE Photonics Technology Letters": "IEEE_Photonics_Technology_Letters",
    "IEEE Photonics Journal": "IEEE_Photonics_Journal",
    "Journal of Lightwave Technology": "J_Lightwave_Technology",
    "Laser and Photonics Reviews": "Laser_Photonics_Reviews",
    "Acta Materialia": "Acta_Materialia",
    "Additive Manufacturing": "Additive_Manufacturing",
    "Virtual and Physical Prototyping": "Virtual_Physical_Prototyping",
    "International Materials Reviews": "Int_Materials_Reviews",
    "Materials & Design": "Materials_Design",
    "Materials and Design": "Materials_Design",
    "Materials Characterization": "Materials_Characterization",
    "Materials Science and Engineering: A": "Mater_Sci_Engineering_A",
    "Intermetallics": "Intermetallics",
    "Metals": "Metals",
    "Metals and Materials International": "Metals_Materials_International",
    "Journal of Magnesium and Alloys": "J_Magnesium_Alloys",
    "Journal of Nuclear Materials": "J_Nuclear_Materials",
    "Nuclear Materials and Energy": "Nuclear_Materials_Energy",
    "Journal of the European Ceramic Society": "J_European_Ceramic_Society",
    "Journal of the American Ceramic Society": "J_American_Ceramic_Society",
    "Journal of Non-Crystalline Solids": "J_Non_Crystalline_Solids",
    "Journal of Applied Crystallography": "J_Applied_Crystallography",
    "Crystal Growth & Design": "Crystal_Growth_Design",
    "CrystEngComm (Print)": "CrystEngComm",
    "Journal of Crystal Growth": "J_Crystal_Growth",
    "Nanomaterials (Print)": "Nanomaterials",
    "Materials (Print)": "Materials",
    "Energies (Print)": "Energies",
    "Molecules (Print)": "Molecules",
    "Polymers (Print)": "Polymers",
    "Nanoscale (Print)": "Nanoscale",
    "Nanotechnology (Print)": "Nanotechnology",
    "Small (Print)": "Small",
    "Advanced Materials (Print)": "Advanced_Materials",
    "Angewandte Chemie (Print)": "Angewandte_Chemie",
    "Journal of the American Chemical Society (Print)": "JACS",
    "Nano Letters (Print)": "Nano_Letters",
    "ACS Nano (Print)": "ACS_Nano",
    "ACS Energy Letters (Print)": "ACS_Energy_Letters",
    "Joule (Print)": "Joule",
    "Nature (Print)": "Nature",
    "Science (Print)": "Science",
    "RSC Advances (Print)": "RSC_Advances",
    "Chemical Communications (Print)": "Chemical_Communications",
    "Chemical Science (Print)": "Chemical_Science",
    "Materials Horizons (Print)": "Materials_Horizons",
    "Physical Chemistry Chemical Physics (Print)": "PCCP",
    "ACS Applied Materials & Interfaces (Print)": "ACS_Applied_Materials_Interfaces",
    "ACS Applied Materials and Interfaces (Print)": "ACS_Applied_Materials_Interfaces",
    "ACS Applied Energy Materials (Print)": "ACS_Applied_Energy_Materials",
    "Chemistry of Materials (Print)": "Chemistry_of_Materials",
    "ACS Omega (Print)": "ACS_Omega",
    "Solar RRL (Print)": "Solar_RRL",
    "Advanced Energy Materials (Print)": "Advanced_Energy_Materials",
    "Advanced Functional Materials (Print)": "Advanced_Functional_Materials",
    "Advanced Science (Print)": "Advanced_Science",
    "Small Methods (Print)": "Small_Methods",
    "Journal of Materials Chemistry A (Print)": "J_Materials_Chemistry_A",
    "Journal of Materials Chemistry C (Print)": "J_Materials_Chemistry_C",
    "Energy & Environmental Science (Print)": "Energy_Environmental_Science",
    "Nanoscale Horizons (Print)": "Nanoscale_Horizons",
    "Nanoscale Advances (Print)": "Nanoscale_Advances",
    "Materials Advances (Print)": "Materials_Advances",
    "Journal of Physical Chemistry Letters (Print)": "J_Physical_Chemistry_Letters",
    "The Journal of Physical Chemistry Letters (Print)": "J_Physical_Chemistry_Letters",
    "Journal of Physics D: Applied Physics (Print)": "J_Physics_D_Applied_Physics",
    "Nanotechnology (Bristol. Print)": "Nanotechnology",

    # ── arXiv / Preprint ──
    "arXiv": "arXiv",
    "arXiv.org": "arXiv",
    "Social Science Research Network": "SSRN",
}

# ── 额外清理: venue 中的城市/国家注释 ──
_LOCATION_PARENS = re.compile(
    r'\s*\((?:Amsterdam|Stuttgart|Bristol|Cambridge|London|Berlin|'
    r'New York|Oxford|Weinheim|Washington|Lausanne|Beijing|'
    r'Tokyo|Singapore|Basel|Netherlands|Germany|UK|USA|'
    r'Mass|Massachusetts|Calif|California)[^)]*\)',
    re.IGNORECASE,
)

# ── 出版商标注 ── (去掉不影响期刊名识别的出版商标注)
_PUBLISHER_PARENS = re.compile(
    r'\s*\((?:Nature Publishing Group|NPG|Nature Research|Nature|'
    r'Springer|Elsevier|Wiley|RSC|ACS|IOP|AIP|APS|MDPI|IEEE|'
    r'Nature Portfolio)[^)]*\)',
    re.IGNORECASE,
)

# ── 缩写展开 ──
_ABBREV_MAP = {
    "nat.": "nature",
    "commun.": "communications",
    "mater.": "materials",
    "nanotechnol.": "nanotechnology",
    "photon.": "photonics",
    "catal.": "catalysis",
    "synth.": "synthesis",
    "sustain.": "sustainability",
    "rev.": "reviews",
    "chem.": "chemistry",
    "phys.": "physics",
    "electron.": "electronics",
    "comput.": "computational",
    "sci.": "science",
    "adv.": "advances",
    "mater": "materials",
    "sci": "science",
    "technol.": "technology",
    "eng.": "engineering",
    "lett.": "letters",
    "appl.": "applied",
    "quant.": "quantum",
}


# ── 公开 API ──

def classify_venue(venue: str) -> str:
    """期刊名 → 安全目录名。

    Args:
        venue: 原始期刊名 (e.g., "ACS Energy Letters",
               "Nano letters (Print)", "Energy &amp; Environmental Science")

    Returns:
        安全目录名 (e.g., "ACS_Energy_Letters", "Nano_Letters")
    """
    if not venue or not venue.strip():
        return "Unknown"

    # 1. 规范化
    cleaned = _normalize_venue(venue)

    if not cleaned:
        return "Unknown"

    # 2. 精确匹配 (原始 → 规范化后 → 去括号)
    result = _exact_lookup(venue) or _exact_lookup(cleaned)
    if not result:
        # 去掉所有括号再试
        no_parens = re.sub(r'\s*\([^)]*\)', '', cleaned).strip()
        result = _exact_lookup(no_parens)
    if result:
        return result

    # 3. 最长 key 优先子串匹配 (规范化后的 venue)
    result = _longest_key_match(cleaned)
    if result:
        return result

    # 4. 激进规范化 (展开缩写) 再试精确匹配
    fuzzy = _normalize_for_matching(venue)
    if fuzzy and fuzzy != cleaned.lower():
        result = _exact_lookup(fuzzy)
        if not result:
            result = _longest_key_match(fuzzy)
        if result:
            return result

    # 5. Fallback: 安全化 venue 名
    return _safe_fallback_dirname(cleaned)


def _normalize_venue(venue: str) -> str:
    """规范化 venue 字符串。"""
    v = venue.strip()

    # HTML 实体解码
    v = _html.unescape(v)

    # 去掉 "(Print)" 后缀
    v = re.sub(r'\s*\(Print\)\s*', ' ', v)

    # 去掉出版商标注: "(Nature Publishing Group)" 等
    v = _PUBLISHER_PARENS.sub('', v)

    # 去掉城市/国家注释: "Optical materials (Amsterdam)" → "Optical materials"
    v = _LOCATION_PARENS.sub('', v)

    # 规范化空白
    v = re.sub(r'\s+', ' ', v).strip()

    # 规范化 &
    v = v.replace('&amp;', '&').replace('&#38;', '&')

    return v


def _normalize_for_matching(venue: str) -> str:
    """为匹配做更激进的规范化: 展开缩写, 去所有括号。

    仅在 _exact_lookup 和 _longest_key_match 都失败后使用。
    """
    v = _normalize_venue(venue)
    v = v.lower()

    # 展开缩写 (按长度降序, 防短词误替换)
    for abbr, full in sorted(_ABBREV_MAP.items(), key=lambda x: -len(x[0])):
        v = v.replace(abbr, full)

    # 去掉所有残留括号内容
    v = re.sub(r'\s*\([^)]*\)', '', v)
    v = re.sub(r'\s+', ' ', v).strip()

    return v


def _exact_lookup(venue: str) -> str | None:
    """精确匹配 (尝试原始值和各种变体)。"""
    v = venue.strip()

    # 直接查表
    if v in EXACT_VENUE_MAP:
        return EXACT_VENUE_MAP[v]

    # 去掉末尾括号后查表: "Reports on Progress in Physics (Print)" → match
    # (已经在 _normalize_venue 中处理, 但这里做二次保证)
    without_parens = re.sub(r'\s*\([^)]*\)\s*$', '', v).strip()
    if without_parens != v and without_parens in EXACT_VENUE_MAP:
        return EXACT_VENUE_MAP[without_parens]

    # 小写查表
    lower_map = {k.lower(): v for k, v in EXACT_VENUE_MAP.items()}
    if v.lower() in lower_map:
        return lower_map[v.lower()]

    if without_parens.lower() in lower_map:
        return lower_map[without_parens.lower()]

    return None


def _longest_key_match(cleaned_venue: str) -> str | None:
    """用最长 key 做子串匹配, 防止短 key 通配。

    例如: "science" 不会匹配 "Science Advances" 因为
    "science advances" (17 chars) 比 "science" (7 chars) 更长且匹配。
    """
    v_lower = cleaned_venue.lower()

    # 按 key 长度降序排列
    sorted_keys = sorted(EXACT_VENUE_MAP.keys(), key=lambda k: -len(k))

    for key in sorted_keys:
        key_lower = key.lower()
        if key_lower in v_lower and _word_boundary_match(key_lower, v_lower):
            return EXACT_VENUE_MAP[key]

    return None


def _word_boundary_match(key: str, text: str) -> bool:
    """检查 key 是否在 text 中作为词边界出现。"""
    idx = text.find(key)
    if idx == -1:
        return False
    # 左边边界
    left_ok = idx == 0 or not text[idx - 1].isalnum()
    # 右边边界
    right_ok = (idx + len(key) == len(text)
                or not text[idx + len(key)].isalnum())
    return left_ok and right_ok


def _safe_fallback_dirname(venue: str) -> str:
    """将任意 venue 转为安全目录名。"""
    # 去掉特殊字符
    safe = re.sub(r'[^\w\s-]', '', venue)
    safe = re.sub(r'\s+', '_', safe)
    # 去重下划线
    safe = re.sub(r'_+', '_', safe).strip('_')
    if not safe:
        return "Unknown"
    return safe[:60]  # 从 40 放宽到 60


# ── 兼容别名 ──

# 让原有代码中的 journal_to_dirname 函数名可以继续使用
journal_to_dirname = classify_venue


# ── 测试 ──

if __name__ == "__main__":
    test_cases = [
        ("Nature", "Nature"),
        ("Nature Communications", "NatComm"),
        ("Nature Energy", "NatEnergy"),
        ("Science", "Science"),
        ("Science Advances", "Science_Advances"),
        ("Energy & Environmental Science", "Energy_Environmental_Science"),
        ("Energy &amp; Environmental Science", "Energy_Environmental_Science"),
        ("Joule", "Joule"),
        ("ACS Energy Letters", "ACS_Energy_Letters"),
        ("Nano letters (Print)", "Nano_Letters"),
        ("Nano Letters", "Nano_Letters"),
        ("Optical materials (Amsterdam)", "Optical_Materials"),
        ("Optical Materials", "Optical_Materials"),
        ("Physical Chemistry, Chemical Physics - PCCP", "PCCP"),
        ("Physical Chemistry Chemical Physics", "PCCP"),
        ("Reports on Progress in Physics (Print)", "Reports_on_Progress_in_Physics"),
        ("Journal of Physics D: Applied Physics", "J_Physics_D_Applied_Physics"),
        ("ACS Applied Materials and Interfaces", "ACS_Applied_Materials_Interfaces"),
        ("ACS Applied Materials & Interfaces", "ACS_Applied_Materials_Interfaces"),
        ("JPhysD: Applied Physics", "J_Physics_D_Applied_Physics"),
        ("Heliyon", "Heliyon"),
        ("Physica Scripta", "Physica_Scripta"),
        ("Scientific Reports", "Scientific_Reports"),
        ("Solar Energy", "Solar_Energy"),
        ("Journal of the American Chemical Society", "JACS"),
        ("Journal of the American Chemical Society (Print)", "JACS"),
        ("", "Unknown"),
        ("Some Random Unknown Journal Name", "Some_Random_Unknown_Journal_Name"),
        ("Innovation (Cambridge (Mass.))", "The_Innovation"),
        ("Sustainable Energy Technologies and Assessments", "Sustainable_Energy_Technologies_Assessments"),
        ("Journal of Materials Science: Materials in Electronics", "J_Materials_Science_Mater_Electron"),
        ("Cell Reports Physical Science", "Cell_Reports_Physical_Science"),
        ("EES Solar", "EES_Solar"),
        ("Materials Today Proceedings", "Materials_Today_Proceedings"),
        ("Solar Energy Materials and Solar Cells", "Solar_Energy_Materials_Solar_Cells"),
        ("Renewable and Sustainable Energy Reviews", "Renewable_Sustainable_Energy_Reviews"),
        ("Renewable & Sustainable Energy Reviews", "Renewable_Sustainable_Energy_Reviews"),
    ]

    all_ok = True
    for venue, expected in test_cases:
        result = classify_venue(venue)
        status = "✅" if result == expected else "❌"
        if result != expected:
            all_ok = False
            print(f"{status} {venue[:50]:<50s} → {result}  (expected: {expected})")

    if all_ok:
        print("All tests passed! ✅")
    else:
        print(f"\n{sum(1 for v,e in test_cases if classify_venue(v)!=e)} failures")
