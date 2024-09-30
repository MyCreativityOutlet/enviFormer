"""
EnviFormer a transformer based method for the prediction of biodegradation products and pathways
Copyright (C) 2024  Liam Brydon
Contact at: lbry121@aucklanduni.ac.nz

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
from enviPath_python.enviPath import *
from enviPath_python.objects import *
import json
import os
from tqdm import tqdm


def download_envipath(save_path, force_download=False):
    host = 'https://envipath.org/'
    bbd_package = f'{host}package/32de3cf4-e3e6-4168-956e-32fa5ddb0ce1'
    soil_package = f'{host}package/5882df9c-dae1-4d80-a40e-db4724271456'
    sludge_package = f'{host}package/7932e576-03c7-4106-819d-fe80dc605b8a'
    eP = enviPath(host)
    packages = [(soil_package, "soil"), (bbd_package, "bbd"), (sludge_package, "sludge")]
    for package, name in packages:
        print(f"Starting {name} download.")
        local_package_path = os.path.join(save_path, f"{name}.json")
        if os.path.exists(local_package_path) and not force_download:
            print(f"Package {name} has already been downloaded.")
            continue
        envi_package = Package(eP.requester, id=package)
        compounds = envi_package.get_compounds()
        reactions = envi_package.get_reactions()
        if name == "soil":
            soil_rule_package = f'{host}package/7983cb64-f580-42ba-8776-8fdace4ad7dc'
            envi_soil_rule = Package(eP.requester, id=soil_rule_package)
            rules = envi_soil_rule.get_rules()
        else:
            rules = envi_package.get_rules()
        pathways = envi_package.get_pathways()
        data_dict = {"compounds": [], "reactions": [], "rules": [], "pathways": []}
        for compound in tqdm(compounds, desc="Getting compounds"):
            data_dict["compounds"].append(compound.get_json())
        for reaction in tqdm(reactions, desc="Getting reactions"):
            data_dict["reactions"].append(reaction.get_json())
        for rule in tqdm(rules, desc="Getting rules"):
            if type(rule) is SimpleRule:
                rule = rule.get_json()
                data_dict["rules"].append(rule)
        for pathway in tqdm(pathways, desc="Getting pathways"):
            data_dict["pathways"].append(pathway.get_json())
        with open(local_package_path, 'w') as data_file:
            json.dump(data_dict, data_file, indent=4)


def check_envipath_data():
    envipath_data_path = "data/envipath"
    if any(not os.path.exists(f"data/envipath/{package}.json") for package in ["soil", 'bbd', 'sludge']):
        os.makedirs(envipath_data_path, exist_ok=True)
        download_envipath(envipath_data_path)
    return
