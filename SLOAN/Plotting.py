# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# %%
COLOR1 = (1,0,0)
COLOR2 = (0,0,1)
C2 = 'blue'
C1 = 'red'

# Plotting the histogram as subplots in columns
fig = plt.figure(figsize=(72, 48))
# Create the outer subplot grid (9x1)
outer_grid = gridspec.GridSpec(9, 4)
# Common yticks
common_yticks = np.linspace(0, 1, 5)

for i in range(9):
    inner_grid = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_grid[i,2])
    ax = plt.subplot(inner_grid[0])
    ind = np.load('4Metrics/Niharika Gonella_Astha Pahwa_0/Directionalhistogram/Directional_Histogram_ind' + str(i) + '0.npy', allow_pickle=True)
    listcontaining = np.load('4Metrics/Niharika Gonella_Astha Pahwa_0/Directionalhistogram/Directional_Histogram_listcontaining' + str(i) + '0.npy', allow_pickle=True)
    width = np.load('4Metrics/Niharika Gonella_Astha Pahwa_0/Directionalhistogram/Directional_Histogram_width' + str(i) + '0.npy', allow_pickle=True)

    ind_width = np.load('4Metrics/Niharika Gonella_Astha Pahwa_0/Directionalhistogram/Directional_Histogram_ind+width' + str(i) + '1.npy', allow_pickle=True)
    listcontaining1 = np.load('4Metrics/Niharika Gonella_Astha Pahwa_0/Directionalhistogram/Directional_Histogram_listcontaining1' + str(i) + '1.npy', allow_pickle=True)
    width = np.load('4Metrics/Niharika Gonella_Astha Pahwa_0/Directionalhistogram/Directional_Histogram_width' + str(i) + '1.npy', allow_pickle=True)

    bars = ('Front', 'Left', 'Back', 'Right')
    y_pos = np.arange(4)

    norm_fac = max([listcontaining.max(), listcontaining1.max()])
    if i == 8:
        ax.bar(ind,listcontaining/norm_fac,width,color=COLOR1)#blue
    else:
        ax.bar(ind,listcontaining/norm_fac,width,color=COLOR1)
    plt.xticks([])

    if i == 8:
        ax.bar(ind_width,listcontaining1/norm_fac,width,color=COLOR2)
    else:
        ax.bar(ind_width,listcontaining1/norm_fac,width,color=COLOR2)
    
    if i == 8:
        ax.set_xticks(y_pos, bars, color='black', rotation=0, fontweight='bold', fontsize=70, horizontalalignment='center')
        ax.set_xlabel('Movement Directions', fontweight='bold', fontsize=80)
    else:
        plt.xticks([])
    if i == 0:
        ax.set_title('Directional Histogram',fontweight='bold', fontsize=80)
    plt.yticks(common_yticks, fontsize=60, fontweight='bold')

for i in range(9):
    eng_dis = np.load('4Metrics/Niharika Gonella_Astha Pahwa_0/EngageDisengage/Engage-Disengage' + str(i) + '.npy', allow_pickle=True)
    inner_grid = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer_grid[i,1])
    ax = plt.subplot(inner_grid[0])
    ax.bar(eng_dis[0][0], eng_dis[0][1],color=COLOR1)
    plt.yticks([])
    plt.xticks([])
    if i == 0:
        ax.set_title('Engage/Disengage',fontweight='bold', fontsize=80)
    ax = plt.subplot(inner_grid[1])
    ax.bar(eng_dis[1][0], eng_dis[1][1],color=COLOR2)
    plt.yticks([])
    plt.xticks([])
    ax = plt.subplot(inner_grid[2])
    ax.bar(eng_dis[2][0], eng_dis[2][1],color='blueviolet')
    plt.yticks([])
    if i == 8:
        ax.set_xlabel('Time in Seconds', fontweight='bold', fontsize=80)
        plt.xticks(np.linspace(0, 160, 3), fontsize=60, fontweight='bold')
    else:
        plt.xticks(np.linspace(i*20, (i+1)*20, 3), fontsize=60, fontweight='bold')

img = np.load('clean_image.npy', allow_pickle=True)
for i in range(9):
    person1 = np.load('4Metrics/Niharika Gonella_Astha Pahwa_0/Hotspot/Complete_Hotspot' + str(i) + '0.npy', allow_pickle=True)
    person2 = np.load('4Metrics/Niharika Gonella_Astha Pahwa_0/Hotspot/Complete_Hotspot' + str(i) + '1.npy', allow_pickle=True)
    inner_grid = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_grid[i,0])
    ax = plt.subplot(inner_grid[0,0])
    ax.imshow(img)
    for kl in range(len(person1)):
        POL=person1[kl]
        if len(POL) == 0:
            print('Detection miss')
        else:
            Act=POL
            bx = round(int(Act[1]))
            by = round(int(Act[2]))
            bw = round(int(Act[3]))
            bh = round(int(Act[4]))	
            Actx=round((bx+bw)/2)
            Acty=round((by+bh)/2)
            circle = plt.Circle((Actx,Acty),5, fc=C1,ec=C1,fill=True)
            plt.gca().add_patch(circle)
    plt.xticks([])
    plt.yticks([])
    if i == 0:
        ax.set_title('Hotspot: P1',fontweight='bold', fontsize=80)
    if i!=8:
        plt.xlabel(str(20*i) + ' to ' + str((i+1)*20) + 'secs',fontweight='bold', fontsize=70)
    else:
        plt.xlabel(str(20*0) + ' to ' + str(8*20) + ' secs',fontweight='bold', fontsize=70)
    ax = plt.subplot(inner_grid[0,1])
    ax.imshow(img)
    for kl in range(len(person2)):
        POL=person2[kl]
        if len(POL) == 0:
            print('Detection miss')
        else:
            Act=POL
            bx = round(int(Act[1]))
            by = round(int(Act[2]))
            bw = round(int(Act[3]))
            bh = round(int(Act[4]))	
            Actx=round((bx+bw)/2)
            Acty=round((by+bh)/2)
            circle = plt.Circle((Actx,Acty),5, fc=C2,ec=C2,fill=True)
            plt.gca().add_patch(circle)
    plt.xticks([])
    plt.yticks([])
    if i == 0:
        ax.set_title('Hotspot: P2',fontweight='bold', fontsize=80)
    if i!=8:
        plt.xlabel(str(20*i) + ' to ' + str((i+1)*20) + 'secs',fontweight='bold', fontsize=70)
    else:
        plt.xlabel(str(20*0) + ' to ' + str(8*20) + ' secs',fontweight='bold', fontsize=70)

for i in range(9):
    inner_grid = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_grid[i,3])
    ax = plt.subplot(inner_grid[0])

    ind = np.load('4Metrics/Niharika Gonella_Astha Pahwa_0/Zonemanagment/Zonemanagment_ind' + str(i) + '0.npy', allow_pickle=True)
    X1 = np.load('4Metrics/Niharika Gonella_Astha Pahwa_0/Zonemanagment/Zonemanagment_X1' + str(i) + '0.npy', allow_pickle=True)
    width = np.load('4Metrics/Niharika Gonella_Astha Pahwa_0/Zonemanagment/Zonemanagment_width' + str(i) + '0.npy', allow_pickle=True)

    ind_width = np.load('4Metrics/Niharika Gonella_Astha Pahwa_0/Zonemanagment/Zonemanagment_ind+width' + str(i) + '1.npy', allow_pickle=True)
    X2 = np.load('4Metrics/Niharika Gonella_Astha Pahwa_0/Zonemanagment/Zonemanagment_X2' + str(i) + '1.npy', allow_pickle=True)
    width = np.load('4Metrics/Niharika Gonella_Astha Pahwa_0/Zonemanagment/Zonemanagment_width' + str(i) + '1.npy', allow_pickle=True)

    norm_fac = max([X1.max(), X2.max()])
    if i == 8:
        ax.bar(ind[0:6],X1[0:6]/norm_fac,width, color=COLOR1)#blue
    else:
        ax.bar(ind[0:6],X1[0:6]/norm_fac,width, color=COLOR1)
    if  i == 8:
        ax.bar(ind_width[0:6],X2[0:6]/norm_fac,width,color=COLOR2)
    else:
        ax.bar(ind_width[0:6],X2[0:6]/norm_fac,width,color=COLOR2)
    if i == 8:
        ax.set_xticks(ind_width[0:6] + width/ 2, ('Z1', 'Z2', 'Z3','Z4','Z5', 'Z6'),fontweight='bold', fontsize=70, horizontalalignment='center')
        ax.set_xlabel('Zone Numbers', fontweight='bold', fontsize=80)
    else:
        plt.xticks([])
    if i == 0:
        ax.set_title('Zone Management',fontweight='bold', fontsize=80)
    plt.yticks(common_yticks, fontsize=60, fontweight='bold')

plt.tight_layout()
plt.savefig('Demonstration_Figure' + '.pdf', dpi = 300)
