// Local modification of tokenizers 0.22.2: compact indexed UTF-8 prefix storage.
const NONE: usize = usize::MAX;
const DENSE_AFTER: u16 = 8;

#[derive(Default)]
pub struct TrieBuilder { trie: Trie }
impl TrieBuilder {
    pub fn push(&mut self, element: &[u8]) { self.trie.push(element); }
    pub fn build(self) -> Trie { self.trie }
}

#[derive(Clone)]
pub struct Trie {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    // UTF-8 has a bounded byte alphabet. Branching nodes use direct byte
    // lookup; short edge lists avoid allocating a table for every prefix.
    tables: Vec<Box<[usize; 256]>>,
}

impl Trie {
    #[inline]
    fn child(&self, node: usize, label: u8) -> Option<usize> {
        let node = &self.nodes[node];
        if node.table != NONE {
            let child = self.tables[node.table][label as usize];
            return (child != NONE).then_some(child);
        }
        let mut edge = node.first;
        while edge != NONE && self.edges[edge].label != label { edge = self.edges[edge].next; }
        (edge != NONE).then(|| self.edges[edge].child)
    }

    pub fn push(&mut self, element: &[u8]) {
        let mut node = 0;
        for label in element {
            node = match self.child(node, *label) {
                Some(child) => child,
                None => {
                    let child = self.nodes.len();
                    self.nodes.push(Node::default());
                    self.nodes[node].children += 1;
                    let table = self.nodes[node].table;
                    if table != NONE {
                        self.tables[table][*label as usize] = child;
                    } else {
                        let next = self.nodes[node].first;
                        self.nodes[node].first = self.edges.len();
                        self.edges.push(Edge { label: *label, child, next });
                        if self.nodes[node].children == DENSE_AFTER {
                            let mut table = Box::new([NONE; 256]);
                            let mut edge = self.nodes[node].first;
                            while edge != NONE {
                                let value = &self.edges[edge];
                                table[value.label as usize] = value.child;
                                edge = value.next;
                            }
                            self.nodes[node].table = self.tables.len();
                            self.tables.push(table);
                        }
                    }
                    child
                }
            };
        }
        self.nodes[node].is_leaf = true;
    }

    pub fn common_prefix_search<T>(&self, iterator: T) -> TrieIterator<'_, T>
    where T: Iterator<Item = u8> {
        TrieIterator { trie: self, node: 0, prefix: vec![], iterator }
    }
}

pub struct TrieIterator<'a, T> {
    trie: &'a Trie,
    node: usize,
    prefix: Vec<u8>,
    iterator: T,
}
impl<T: Iterator<Item = u8>> Iterator for TrieIterator<'_, T> {
    type Item = Vec<u8>;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let label = self.iterator.next()?;
            self.prefix.push(label);
            self.node = self.trie.child(self.node, label)?;
            if self.trie.nodes[self.node].is_leaf { return Some(self.prefix.clone()); }
        }
    }
}

impl Default for Trie {
    fn default() -> Self {
        Self {
            nodes: vec![Node { table: 0, ..Node::default() }],
            edges: Vec::new(), tables: vec![Box::new([NONE; 256])],
        }
    }
}

#[derive(Clone)]
pub struct Node { is_leaf: bool, first: usize, table: usize, children: u16 }
#[derive(Clone)]
struct Edge { label: u8, child: usize, next: usize }
impl Default for Node {
    fn default() -> Self { Self { is_leaf: false, first: NONE, table: NONE, children: 0 } }
}
