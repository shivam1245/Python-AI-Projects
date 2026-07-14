/* ─── Overview page: Mobile Sidebar ──────────────────────────────── */
const sidebar   = document.getElementById('sidebar');
const hamburger = document.getElementById('hamburger');
const overlay   = document.getElementById('overlay');

if (sidebar && hamburger && overlay) {
  const openSidebar  = () => { sidebar.classList.add('open'); hamburger.classList.add('open'); overlay.classList.add('visible'); document.body.style.overflow = 'hidden'; };
  const closeSidebar = () => { sidebar.classList.remove('open'); hamburger.classList.remove('open'); overlay.classList.remove('visible'); document.body.style.overflow = ''; };

  hamburger.addEventListener('click', () => sidebar.classList.contains('open') ? closeSidebar() : openSidebar());
  overlay.addEventListener('click', closeSidebar);
  document.querySelectorAll('.nav-link').forEach(l => l.addEventListener('click', () => { if (window.innerWidth <= 900) closeSidebar(); }));
}

/* ─── Scroll Spy ──────────────────────────────────────────────────── */
const spyObserver = new IntersectionObserver(entries => {
  entries.forEach(e => {
    if (!e.isIntersecting) return;
    document.querySelectorAll('.nav-link[data-section]').forEach(l => {
      l.classList.toggle('active', l.dataset.section === e.target.id);
    });
  });
}, { rootMargin: '-30% 0px -60% 0px', threshold: 0 });

document.querySelectorAll('section[id]').forEach(s => spyObserver.observe(s));

/* ─── Card Reveal Animations ─────────────────────────────────────── */
const revealObserver = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (!entry.isIntersecting) return;
    const siblings = Array.from(entry.target.parentElement.querySelectorAll('.reveal'));
    const idx = siblings.indexOf(entry.target);
    setTimeout(() => entry.target.classList.add('visible'), idx * 70);
    revealObserver.unobserve(entry.target);
  });
}, { threshold: 0.08 });

document.querySelectorAll('.reveal').forEach(el => revealObserver.observe(el));

/* ─── Counter Animation ──────────────────────────────────────────── */
function animateCount(el, target, duration = 1200) {
  let start = null;
  const step = ts => {
    if (!start) start = ts;
    const p = Math.min((ts - start) / duration, 1);
    el.textContent = Math.round((1 - (1 - p) ** 2) * target);
    if (p < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

const counterObserver = new IntersectionObserver(entries => {
  entries.forEach(e => {
    if (!e.isIntersecting) return;
    e.target.querySelectorAll('.stat-num[data-target]').forEach(el => animateCount(el, +el.dataset.target));
    counterObserver.unobserve(e.target);
  });
}, { threshold: 0.3 });

document.querySelectorAll('.hero-stats').forEach(el => counterObserver.observe(el));

/* ─── Progress Bar Animation ─────────────────────────────────────── */
const progressObserver = new IntersectionObserver(entries => {
  entries.forEach(e => {
    if (!e.isIntersecting) return;
    e.target.querySelectorAll('.progress-fill[style]').forEach(bar => {
      const target = bar.style.width;
      bar.style.width = '0%';
      requestAnimationFrame(() => requestAnimationFrame(() => { bar.style.width = target; }));
    });
    progressObserver.unobserve(e.target);
  });
}, { threshold: 0.2 });

document.querySelectorAll('.card-tier2').forEach(el => progressObserver.observe(el));

/* ─── Smooth scroll with header offset ───────────────────────────── */
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener('click', e => {
    const id = anchor.getAttribute('href').slice(1);
    const target = document.getElementById(id);
    if (!target) return;
    e.preventDefault();
    window.scrollTo({ top: target.getBoundingClientRect().top + window.scrollY - 72, behavior: 'smooth' });
  });
});
