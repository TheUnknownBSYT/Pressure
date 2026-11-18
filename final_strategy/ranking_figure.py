"""Plot conditional match-level uncertainty for each selected player."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def render(players, dest):
    fig, axes = plt.subplots(1,2,figsize=(9.4,4.7),layout='constrained',sharex=True)
    for ax,(role,title) in zip(axes,[('bat','Batters'),('bowl','Bowlers')]):
        z=players[players.role.eq(role)].sort_values('rank')
        y=list(range(len(z),0,-1))
        ax.hlines(y,z.absolute_lo95,z.absolute_hi95,color='#738C83',linewidth=1.8)
        ax.scatter(z.absolute,y,s=24,color='#155544',zorder=3)
        ax.axvline(0,color='#454B48',linestyle='--',linewidth=1)
        ax.set_yticks(y,z.player_name,fontsize=9)
        ax.set_title(title,loc='left',fontsize=12,weight='bold')
        ax.set_xlabel('Runs above peers per six pressure balls',fontsize=9)
        ax.spines[['right','top','left']].set_visible(False)
        ax.tick_params(axis='y',length=0)
        ax.set_ylim(.4,10.65)
    fig.savefig(dest,dpi=190,bbox_inches='tight')
    plt.close(fig)
