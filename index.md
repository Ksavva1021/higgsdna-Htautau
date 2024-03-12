Table of Contents
=================

* [1. Introduction](#1-introduction)
* [2. Columnar Analysis](#2-columnar)
  * [2.1 Columnar vs. Per-Event Analysis](#21-columnar-vs-per-event-analysis)
  * [2.2 Complex selections in columnar-style](#22-complex-selections-in-columnar-style)
  * [2.3 Notes on awkward, uproot, and vector](#23-notes-on-awkward-uproot-and-vector)
* [3. Setup](#3-setup)
  * [3.1 Environment](#31-environment)
  * [3.2 Installation](#32-installation)
* [4. Main Concepts](#4-main-concepts)
  * [4.1 Command Line Tool](#41-command-line-tool)
  * [4.2 Processor](#42-processor)
    * [4.2.1 Coffea Processor](#421-coffea-processor)
    * [4.2.2 HtautauBaseProcessor](#422-HtautauBaseProcessor)
* [5. Workflow - Analysis Chain](#5-workflow--analysis-chain)
  
----

# 1. Introduction
Welcome to the documentation record of the Higgs to tau tau version of the HiggsDNA framework. HiggsDNA provides tools for developing and executing Higgs to ditau analyses, starting from the nanoAOD data tier.

# 2 Columnar Analysis
HiggsDNA is based on a "columnar" style of analysis, where event selections are performed in a vectorized fashion (i.e. `numpy`-style operations).
One of the core dependencies of HiggsDNA is `awkward`, which is very similar to `numpy` in terms of user interface, but has the advantage of providing superior performance on "jagged" arrays.
A jagged array has one or more dimensions which are variable in length. In a HEP-context, this shows up quite frequently: for example, the number of hadronic jets in a given event may be 2, 11, or even 0.
`numpy` is not optimized to handle jagged arrays, while `awkward` is.

## 2.1 Columnar vs. Per-Event Analysis
Before diving into tools further, lets compare "columnar" analysis with "per-event" analysis, which is more traditionally used in HEP.
In "per-event" analysis, one explicitly performs loops through the events and objects in each event.
For example, consider a dummy analysis where we select for events with 3 or more jets, making some basic requirements on each jet.
In the "per-event" style:
```python
selected_events = []
for evt in events:
    n_jets = 0
    for jet in evt.Jet:
        if jet.pt < 25:
            continue
        if abs(jet.eta) < 2.4:
            continue
        n_jets += 1
    if n_jets >= 3:
        selected_events.append(evt)
```
vs. in the "columnar" style with `awkward`:
```
import awkward
jet_cut = (events.Jet.pt > 25.) & (abs(events.Jet.eta) < 2.4) 
selected_jets = events.Jet[jet_cut]

cut = awkward.num(selected_jets) >= 3
selected_events = events[cut]
```

Why should we prefer the columnar-style to the per-event-style?
The reason is that loops like the one shown above are extremely slow in `python`, due to the fact that `python` is not a compiled language and performs type-checking at each point in the loop.
While the columnar code above is not explicitly compiled, its operations are, and this gives us a substantial performance boost.
We will also see that the per-event style can still be used without the performance loss by using the module `numba`, which can perform compilation of `python` functions.

## 2.2 Complex selections in columnar-style
For something like counting the number of jets, it is simple enough to see how to translate per-event-style code into columnar-style code.
However, what if we are doing something more complicated, like creating Z candidates out of opposite-sign same-flavor (OSSF) lepton pairs?
Supposing we have already created arrays of our selected electrons and selected muons, we could construct OSSF lepton pairs as follows.
First, we will import the scikit-hep `vector` package, which allows us to perform four-vector operations and works nicely with `awkward`.
```python
import awkward
import vector
vector.register_awkward()
```
the `register_awkward()` line registers `awkward.Array` behaviors globally with `vector` so that when can identify certain `awkward` arrays as four vectors and have the operations overloaded as we expect (along with other functions and properties, like `.mass` or `.deltaR()`).
Back to our example:
```python
electrons = awkward.Array(electrons, with_name = "Momentum4D") # now we can do things like electrons.deltaR(photons)
ele_pairs = awkward.combinations(
    electrons, # objects of make combinations out of
    2, # how many objects in each combination
    fields = ["LeadLepton", "SubleadLepton"] # can access these as e.g. ee_pairs.LeadLepton.pt
)
```
we have now created all possible pairs of 2 electrons in each event. We could do the same for muons and then concatenate these arrays together:
```python
muon_pairs = awkward.combinations(muons, 2, fields = ["LeadLepton", "SubleadLepton"])
dilep_pairs = awkward.concatenate(
    [ele_pairs, muon_pairs], # arrays to concatenate
    axis = 1 # this keeps the number of events constant, and increases the number of pairs per event. axis = 0 would increase the number of events
)
dilep_pairs["ZCand"] = dilep_pairs.LeadLepton + dilep_pairs.SubleadLepton # these add as 4-vectors since we registered them as "Momentum4D" objects
```
Now we can place some cuts on the z candidates. Lets enforce that they are opposite-sign (same-flavor has already been enforced through construction) and have an invariant mass in the [86, 96] GeV range:
```
os_cut = dilep_pairs.LeadLepton.charge * dilep_pairs.SubleadLepton.charge == -1
mass_cut = (dilep_pairs.ZCand.mass > 86.) & (dilep_pairs.ZCand.mass < 96.)
cut = os_cut & mass_cut
dilep_pairs = dilep_pairs[cut]
```
At this point, we might want to flatten the Z candidates and perform any further analysis on a per-z-candidate basis (rather than per-event basis):
```
dilep_pairs = awkward.flatten(dilep_pairs) # no longer a jagged array
```
or we might want to simply select events that have at least 1 z-candidate:
```
z_candidate_cut = awkward.num(dilep_pairs) >= 1
events = events[z_candidate_cut]
```

If you prefer per-event analysis to columnar analysis, this is also possible in HiggsDNA!
Lets compare an even more complex example and see how this would be done in columnar-style and then in per-event style.
Suppose we want to select jets that are at least deltaR of 0.2 away from all selected leptons in an event.
In columnar style,
```python
def delta_R(objects1, objects2, min_dr):
    """
    Select objects from objects1 which are at least min_dr away from all objects in objects2.
    """
    # Step 1: make sure each of these arrays have at least 1 object for each
    if awkward.count(objects1) == 0 or awkward.count(objects2) == 0: # make sure each of these arrays have at least 1 object for each
        return objects1.pt < 0.

    # Step 2: make sure each array is cast as a four vector
    if not isinstance(objects1, vector.Vector4D):
        objects1 = awkward.Array(objects1, with_name = "Momentum4D")
    if not isinstance(objects2, vector.Vector4D):
        objects2 = awkward.Array(objects2, with_name = "Momentum4D")

    # Step 3: manipulate shapes so they can be cast together
    obj1 = awkward.unflatten(objects1, counts = 1, axis = -1) # shape [n_events, n_obj1, 1]
    obj2 = awkward.unflatten(objects2, counts = 1, axis = 0) # shape [n_events, 1, n_obj2]

    dR = obj1.deltaR(obj2) # shape [n_events, n_obj1, n_obj2]

    # Step 4: select objects from objects1 which are at least deltaR of min_dr away from all objects in objects2
    selection = awkward.all(dR >= min_dr, axis = -1) # shape [n_events, n_obj1]
    return selection
```
and in per-event style, we can use the package `numba` to compile our python loops.
```python
def delta_R_numba(objects1, objects2, min_dr):
    """
    This performs the exact same thing as the function before, but this calls an additional function, `compute_delta_R` which is compiled with `numba`.
    """
    n_objects1 = awkward.num(objects1)
    n_objects2 = awkward.num(objects2)

    offsets, contents = compute_delta_R(
            objects1, n_objects1,
            objects2, n_objects2,
            min_dr
    )

    selection = awkward_utils.construct_jagged_array(offsets, contents)

    return selection

@numba.njit # decorate function with a numba decorator so it gets compiled before it is called
def compute_delta_R(objects1, n_objects1, objects2, n_objects2, min_dr):
    n_events = len(objects1)

    offsets = numpy.zeros(n_events + 1, numpy.int64) # need to tell numba what type these are 
    contents = []

    for i in range(n_events):
        offsets[i+1] = offsets[i] + n_objects1[i]
        for j in range(n_objects1[i]):
            contents.append(True)
            offset_idx = offsets[i] + j
            for k in range(n_objects2[i]):
                if not contents[offset_idx]:
                    continue
                obj1 = vector.obj( # vector works nicely with numba!
                        pt = objects1[i][j].pt,
                        eta = objects1[i][j].eta,
                        phi = objects1[i][j].phi,
                        mass = objects1[i][j].mass
                )
                obj2 = vector.obj(
                        pt = objects2[i][k].pt,
                        eta = objects2[i][k].eta,
                        phi = objects2[i][k].phi,
                        mass = objects2[i][k].mass
                )
                dR = obj1.deltaR(obj2)
                if dR < min_dr:
                    contents[offset_idx] = False

    return offsets, numpy.array(contents)
```

## 2.3 Notes on `awkward`, `uproot`, and `vector`
When loading events from a flat `root` file, `uproot` and `awkward` work together nicely to load the events in an inuitive way:
```python
import uproot
import awkward

f_nano = "root://redirector.t2.ucsd.edu//store/user/hmei/nanoaod_runII/HHggtautau/ttHJetToGG_M125_13TeV_amcatnloFXFX_madspin_pythia8_RunIIFall17MiniAODv2-PU2017_12Apr2018_94X_mc2017_realistic_v14-v1_MINIAODSIM_v0.6_20201021/test_nanoaod_1.root"

with uproot.open(f_nano) as f:
    tree = f["Events"] # can access TTrees by name
    events = tree.arrays(library = "ak", how = "zip")
```
Here, by specifying `library = "ak", how = "zip"`, we are now able to access branches as records, with any jagged-length branches zipped together.
It is easiest to illusrate this through some examples:
```
events["MET_pt"] # access met pt
events.MET_pt # also works to access met pt

events.Jet # jagged array with all jet sub-fields accessible as records
events.Jet.pt # access jet pt
events.Jet.eta # or jet eta
events[("Jet", "pt")] # can also access them through a tupled name
```

In our `events = tree.arrays(...` line we could also specify to only read certain branches from the TTree, which can drastically improve the I/O time:
```
with uproot.open(f_nano) as f:
    tree = f["Events"] # can access TTrees by name
    events = tree.arrays(["MET_pt", "Jet_pt", "Jet_eta"], library = "ak", how = "zip")
```

When adding or modifying fields of an `awkward.Array`, it is important to call them by the string for their name.
For example, if we wanted to place a jet cut and update the jet collection:
```
jet_cut = events.Jet.pt > 25.
events["Jet"] = events.Jet[jet_cut] # works
events["Jet"] = events["Jet"][jet_cut] # also works
events.Jet = events.Jet[jet_cut] # does NOT work

# also true for creating new fields
events["SelectedJet"] = events.Jet[jet_cut] # works
events.SelectedJet = events.Jet[jet_cut] # does NOT work

# and for nested fields
events[("Jet", "pt")] = awkward.ones_like(events.Jet.pt) # works
events.Jet.pt = awkward.ones_like(events.Jet.pt) # does NOT work
events["Jet"].pt = awkward.ones_like(events.Jet.pt) # does NOT work
```
all of the examples above will run without error, but some of them do not do what you intend.

# 3. Setup
This section contains details on how to set up the base environment for HiggsDNA as well as, instructions on how to install HiggsDNA.

## 3.1 Environment
Installing [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html):
```
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
# Linux/bash:
./bin/micromamba shell init -s bash -p ~/micromamba  # this writes to your .bashrc file
# sourcing the bashrc file incorporates the changes into the running session.
# better yet, restart your terminal!
source ~/.bashrc
```
## 3.2 Installation
```
git clone https://gitlab.cern.ch/dwinterb/HiggsDNA.git
cd HiggsDNA
micromamba env create -f environment.yml
micromamba activate higgs-dna
pip install -e .[dev]

# Repository is cloned via https, you can set up a remote via ssh
git remote add origin_SSH ssh://git@gitlab.cern.ch:7999/dwinterb/HiggsDNA.git

# Adding the main HiggsDNA (diphoton) as a remote
git remote add CMS https://gitlab.cern.ch/HiggsDNA-project/HiggsDNA.git
```
# 4. Main Concepts
This section describes some of the main concepts of HiggsDNA.
## 4.1 Command Line Tool
If you want to run an analysis with processors and taggers that have already been developed, the suggested way is to use the command line tool `run_analysis.py`. The main parts that define an analysis are the following:

- `datasets`: Path to a JSON file in the form {"dataset_name": [list_of_files]} (like the one dumped by dasgoclient)
- `workflow`: The coffea processor you want to use to process your data, can be found in the modules located inside the subpackage `higgs_dna.workflows`
- `metaconditions`: Name (without .json extension) of one of the JSON files. This holds selections such as ID choices, MET type, number of leptons/particles for each channel, Met filters, trigger selections, etc. This file is located in `higgs_dna.metaconditions`
- `year`: the year condition
- `taggers`: Set of taggers you want to use, can be found in the modules located inside the subpackage `higgs_dna/workflows/taggers`
- systematics: Set of systematics to be used
- corrections: Set of corrections to be used

These parameters are specified in a JSON file and passed to the command line with the flag `--json-analysis`

```run_analysis.py --json-analysis simple_analysis.json```

where `simple_analysis.json` looks like this:
```
{
    "samplejson": "examples/ditau_analysis/dyzll.json",
    "workflow": "ditau",
    "year": "2022_postEE",
    "metaconditions": "Era2022_postEE_Htautau_v1",
    "taggers": [
    ],
    "systematics": {
    },
    "corrections": {
        ["Pileup","Tau_ID","Tau_ES","Electron_ID","Electron_Reco","Muon_ID","Muon_Isolation","Muon_Reco"]
    }
}
```
where the `taggers` list, `systematics`, and `corrections` dictionaries can be left empty if no taggers or systematics are applied.

The next two flags to be specified are `dump` and `executor`. The former receives the path to a directory where the parquet output files will be stored, while the latter specifies the Coffea executor used to process the chunks of data. The options are the following:

- iterative
- futures
- dask/condor
- dask/slurm
- dask/lpc
- dask/lxplus
- dask/casa
- parsl/slurm
- parsl/condor
- imperial_lx


A few notable options are available with the command line tool are `--chunk`, `---debug`, and `--environment`. The latter is to be used with the `--executor imperial_lx` option such that the job sent to the cluster sources the appropriate environment.

As usual, a description of all the options is printed when running:
```run_analysis.py --help```

## 4.2 Processor
Columnar analysis is a paradigm that describes the way the user writes the analysis application that is best described in contrast to the traditional paradigm in high-energy particle physics (HEP) of using an event loop. In an event loop, the analysis operates row-wise on the input data (in HEP, one row usually corresponds to one reconstructed particle collision event.) Each row is a structure containing several fields, such as the properties of the visible outgoing particles that were reconstructed in a collision event. The analysis code manipulates this structure to either output derived quantities or summary statistics in the form of histograms. In contrast, columnar analysis operates on individual columns of data spanning a chunk (partition, batch) of rows using array programming primitives in turn, to compute derived quantities and summary statistics. Array programming is widely used within the scientific Python ecosystem, supported by the numpy library. However, although the existing scientific Python stack is fully capable of analyzing rectangular arrays (i.e. no variable-length array dimensions), HEP data is very irregular, and manipulating it can become awkward without first generalizing array structure a bit. The awkward package does this, extending array programming capabilities to the complexity of HEP data.

### 4.2.1 Coffea Processor
In almost all HEP analyses, each row corresponds to an independent event, and it is exceptionally rare to need to compute inter-row derived quantities. Due to this, horizontal scale-out is almost trivial: each chunk of rows can be operated on independently. Further, if the output of an analysis is restricted to reducible accumulators such as histograms (abstracted by dask, dask-awkward, and dask-histogram), then outputs can even be merged via tree reduction. The ProcessorABC class is an abstraction to encapsulate analysis code so that it can be easily scaled out, leaving the delivery of input columns and reduction of output accumulators to the coffea framework. However, it is not an absolute requirement and merely a useful organizational framework.

### 4.2.2 HtautauBaseProcessor
Since in the Higgs to ditau analysis some operations are common to several other analyses, a base processor HtautauBaseProcessor was created which can be used in several other basic analyses. Currently, when running the command-line tool the constructor of this class is fed the following:

- `metaconditions`: Name (without .json extension) of one of the JSON files. This holds selections such as ID choices, MET type, number of leptons/particles for each channel, Met filters, trigger selections, etc. This file is located in `higgs_dna.metaconditions`
- `systematics`: List of systematics
- `corrections`: List of corrections
- `apply_trigger`: Boolean to apply triggers (Not used)
- `output_location`: Location to store the output .parquet files
- `taggers`: List of taggers
- `trigger_group`: Not used at the moment but included as an option
- `analysis`: Name of the analysis
- `year`

# 5. Workflow - Analysis Chain
This section will go over the workflow as seen in the base class under the process function. The plan is to update this as more functionality is added. 

The process function is the heart of our analysis code. To begin with, there is a check on whether the dataset is `MC or Data` and the number of effective events is calculated accordingly. If the dataset is MC then, the effective event number is calculated as follows:

`n_effective = n_{o} +ve generator weights - n_{o} of -ve generator weights`

where these quantities are drawn from the genWeight variable stored in NanoAOD for each event.

The metaconditions file is used to define the channels to process but, the channels are set to all channels by default if the metaconditions file does not include a selection. At this point the list of systematics and corrections to process are assigned to a variable. At this point in time, the systematics and corrections are assumed to be event weight based but, this will be updated once corrections and systematics affecting object-level quantities such as $p_{T}$ are implemented.

MET filters are applied to the events via the `apply_met_filters` function which applies MET filters to the events based on a list of filters specified in the metaconditions. The MET filters are applied using the logical AND operation on the individual filter flags. 

The next step is to store the nominal object collections in a dictionary `objs_dct["nominal"] = (original_electrons, original_muons, original_taus)`. These collections are drawn from the corresponding collections in NanoAOD. The same applies for jets and MET. At this point, the need of storing these collections in a dictionary with key set to "nominal" might seem a bit vague but, when systematics affecting the object-level quantities such as $p_{T}$ are implemented, this will be important. This is important because such corrections/systematics do not affect the event as a global weight but, affect which events pass the selections. Unfortunately, there is not a clean way around this and the processing for each one of these systematic variations needs to be run in a loop thus, the need for these keys in the objs_dct. 

Diving into the processing that takes place for each of these variations:

- Muon, Tau, Electron preselections
- For each event, muons, taus, electrons are sorted respectively in $p_{T}$ in descending order
- For each event, muons and electrons are sorted by isolation (ascending)
- For each event, taus are sorted based on the RAW ID score for the vsJet ID, descending (high score --> more isolated)
- Pairs are created for each channel
  - Same-type pairs (mm, ee, tt): Combinations of the objects are made and then the pairs are sorted such that the first object in the pair has the highest $p_{T}$.
  - For mixed channels (em, et, mt): For combinations of mixed pairs we follow the usual HTT conventions and lighter leptons are preferred for obj_1. Note when using the cartesian function the pairs will be ordered preferentially by the objects passed as the first argument. We want to order the pairs prefering muons > electrons > taus.
- Looping over Channels and pairs in each channel:
  -  Applying a $\Delta$R cut > 0.5 between the two objects in the pair
  -  Applying $p_{T}$ and $\eta$ cuts to the objects based on selections in specified in metaconditions
  -  Applying extra lepton vetoes to ensure orthogonality
  -  Only hold the first pair passing the selections (highest ranked one)
  -  Drop None
  -  if the sample is MC, add relevant gen information `add_pair_gen_info`
  -  Adding `pair-level quantities`
  -  Now we pick up the MET and jets, and we apply the same mask that we used to select only our good pairs
  -  First we filter the jets that are matched in dR to our selected tau candidates
  -  compute and add our jet quantities `add_jet_quantities`
  -  For other jets we filter the bjets that are matched in dR to our selected tau candidates
  -  Add MET quantities e.g MET, transverse masses etc `add_met_quantities`
  -  Apply trigger matching which adds booleons to the array indicating whether the pair legs were matched to particular trigger objects
  -  if sample is MC, event-based weights are added (systematics will be added here as well)
  -  Outputs are dumped into a `.parquet` file

