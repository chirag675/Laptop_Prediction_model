import pickle

options = {
    "Company": ['HP','Dell','Lenovo','Apple','Asus','Acer','MSI','Toshiba','Samsung','Razer'],
    "TypeName": ['Notebook','Ultrabook','Gaming','2 in 1 Convertible','Workstation'],
    "Cpu brand": ['Intel','AMD','Other'],
    "Gpu brand": ['Intel','AMD','Nvidia'],
    "os": ['Windows','Mac','Linux','Other']
}

pickle.dump(options, open('options.pkl','wb'))

print("✅ options.pkl created successfully")