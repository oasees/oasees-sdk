# Oasees SDK

## Installation
The SDK is embedded into the Jupyter Notebook image which is built upon the creation of the Oasees Stack. This means that no installation of the SDK and its requirements is needed from the user's side. It only needs to be imported either into a Jupyter Notebook or Python Console by running:
```
import oasees_sdk
```
<br/>

Or in the stack's Jupyter Terminal (all three options are found in the portal's <i>Notebook</i> tab), where you first need to launch the python shell by running:
```
python3
```
And then import the oasees_sdk.

<br/>


## Usage / Instructions
As soon as the SDK gets successfully imported into the chosen python environment, the following brief documentation about its functions and their usage is printed:

```
╔═════════════════════════════════════════════════════════════════════════════════════╗
║ OASEES SDK methods 	 	                                                          ║
║ ------------------------------                                                      ║
║                                                                                     ║
║     my_algorithms() 		                                                          ║
║         Returns a list with all the algorithms purchased from your account          ║
║         on the OASEES Marketplace.                                                  ║
║                                                                                     ║
║                                                                                     ║
║     my_devices() 	   	                                                              ║
║         Returns a list with all the devices purchased / uploaded from your account  ║
║         on the OASEES Marketplace.                                                  ║
║                                                                                     ║
║                                                                                     ║
║     deploy_algorithm(algorithm_title: str) 	 	                                  ║
║         Deploys a purchased algorithm on all your connected devices.                ║
║                                                                                     ║
║         - algorithm_title: Needs to be provided in "string" form.                   ║
║                                                                                     ║
║         e.g. algorithm.py -> deploy_algorithm("algorithm.py")                       ║
║                                                                                     ║
║                                                                                     ║
║                                                                                     ║
║     deploy_local_file(path: str) 	   	                                              ║
║         Deploys the file found in the specified path on all your connected devices. ║
║                                                                                     ║
║         - path: -> Needs to be provided in "string" form.                           ║
║                 -> Is equal to the filename when the file is located in             ║
║                    the Jupyter Notebook's directory.                                ║
║                                                                                     ║
║         e.g. algorithm.py -> deploy_local_file("algorithm.py")                      ║
║                                                                                     ║
║                                                                                     ║
║                                                                                     ║
║     instructions() 	 	                                                          ║
║         Reprints the above documentation.                                           ║
║                                                                                     ║
╚═════════════════════════════════════════════════════════════════════════════════════╝
```