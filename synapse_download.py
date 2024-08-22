import synapseclient 
import synapseutils 
from synapseclient import File, Folder

#To generate a personal access token in the web client, navigate to your Account Settings page, 
# scroll to the bottom of the screen, and click Manage Personal Access Tokens. 
# You can view a list of the existing tokens and their permissions, 
# or click Create New Token to generate a new personal access token with customized access 
# to your Synapse account.



syn = synapseclient.Synapse() 
syn.login(authToken="eyJ0eXAiOiJKV1QiLCJraWQiOiJXN05OOldMSlQ6SjVSSzpMN1RMOlQ3TDc6M1ZYNjpKRU9VOjY0NFI6VTNJWDo1S1oyOjdaQ0s6RlBUSCIsImFsZyI6IlJTMjU2In0.eyJhY2Nlc3MiOnsic2NvcGUiOlsidmlldyIsImRvd25sb2FkIiwibW9kaWZ5Il0sIm9pZGNfY2xhaW1zIjp7fX0sInRva2VuX3R5cGUiOiJQRVJTT05BTF9BQ0NFU1NfVE9LRU4iLCJpc3MiOiJodHRwczovL3JlcG8tcHJvZC5wcm9kLnNhZ2ViYXNlLm9yZy9hdXRoL3YxIiwiYXVkIjoiMCIsIm5iZiI6MTcyNDMzNjczMCwiaWF0IjoxNzI0MzM2NzMwLCJqdGkiOiIxMTI5NyIsInN1YiI6IjMzOTc5MzYifQ.QJjjlQaVsL2aGJmkA6wVDd7bnLM9TogTUYC-ycjOKBUsG84URDvbLzprhZdgzFonnPG_e9AyX5ecFg0NrQ07KNJ5u6J-3vA6fOKP1xUSNMLlqA_hYbCbtP5t21ghu5dBXbOGlyXiKxYBDiSiigiaJ9sR13-SSGx_OlLDdANnICsTW0o8MKTFOMjAD6hJ0gy9-6QjTO5nSdco2HYbRISosF_JlgvsnKp51yVtr3C14gO1xeB6pkpN2UbybjTYQClOEyWWD5jsJQeoE3AKu29bt3Xw-AQQF-REcl9XZI4OWMm7jioDKrWI5gvzZfXDErswLFZylOwHiC_f9LQ2c7rZnQ") 
#files = synapseutils.syncFromSynapse(syn, 'syn21898456')

synapse_id = 'syn21898456'
path_to_save= './HeiCo-2024/'

files = synapseutils.syncFromSynapse(syn,synapse_id, path=path_to_save)
# entity = syn.get(synapse_id, downloadLocation=path_to_save) 

# # List all files and folders within the project
# contents = syn.getChildren(synapse_id)
# # Print the contents
# for item in contents:
#     print(f"{item['type']} named {item['name']} with id {item['id']}")
# #Example: Download all files in the project
# for item in contents:
#     if item['type'] == 'file':  # Only download files
#         file = syn.get(item['id'], downloadLocation=path_to_save)
#         print(f"Downloaded file: {file.path}")

